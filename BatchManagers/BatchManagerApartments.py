import torch.nn.functional as func
from typing import *
import numpy as np
from rich import print as rprint
import random


from Configs import *


class BatchManager():
    def __init__(self,
                 ddm: Union["DDPM", "DDIM"],
                 skip_step: int,
                 device: str,
                 num_epochs: int,
                 batch_size: int,
                 traj_len: int,
                 dataset: torch.utils.data.Dataset,
                 mode: Literal["Shared t", "Consecutive", "Uniform"] = "Uniform"
                 ):
        """
        Initialize the Batch Manager

        :param ddm: The DDPM or DDIM manager
        :param skip_step: How many steps to skip if using DDIM
        :param device: The device to use
        :param batch_size: The batch size
        :param x_shape: The shape of the data, batch size excluded, e.g. [3, 32, 32] for CIFAR-10
        :param dataset: The dataset, should implement __getitem__ and __len__
        """
        self.ddm = ddm
        self.device = device
        self.skip_step = skip_step
        self.B = batch_size
        self.T = ddm.T
        self.L = traj_len
        self.dataset = dataset
        self.mode = mode

        # Compute total iterations
        # subtract 2 because:
        # 1. We need to load batch_size data to fill the batch at the beginning, no training is done
        # 2. The last batch_size data, we cannot have a full batch, so we ignore it
        self.total_iterations = num_epochs * (len(dataset) - 2 * batch_size)

        self.data_idx = 0  # Which data to load next

        # Randomly shuffle the dataset indices, so that we can load data randomly
        self.dataset_idx_mapping = np.random.permutation(len(dataset))

        # Just a tensor {0, 1, ..., B-1} for selecting
        self.batch_idx = torch.arange(self.B, dtype=torch.int32, device=self.device)

        # Create a list of tau values, or skip_step times
        self.tau_list = list(range(self.T - 1, -1, -skip_step))
        if self.tau_list[-1] != 0:
            self.tau_list.append(0)
        rprint(
            f"[red]With T={self.T}, Skip Step={skip_step}, Tau List={self.tau_list[:5]}...{self.tau_list[-5:]}[/red]")
        # Number of tau values
        self.Tau = len(self.tau_list)

        # Compute memory usage for all tensors
        self.tau = torch.zeros(self.B, dtype=torch.int32, device=self.device)
        self.tau_next = torch.zeros(self.B, dtype=torch.int32, device=self.device)
        self.eps_0_to_tp1 = [torch.zeros(self.T, 2, self.L, device=self.device) for _ in range(self.B)]
        self.inputs = torch.zeros(self.B, self.T + 1, 6, self.L, dtype=torch.float32, device=self.device)
        self.masks = torch.zeros(self.B, 3, self.L, dtype=torch.bool, device=self.device)
        self.loc_means = torch.zeros(self.B, 2, 1, dtype=torch.float32, device=self.device)
        self.metas = torch.zeros(self.B, 4, self.L, dtype=torch.long, device=self.device)
        self.s_tp1_to_T = list()


    def registerState(self, shape: List[int]):
        """
        Register a state tensor
        :param name: The name of the state
        :param shape: The shape of the state
        :return:
        """
        self.s_tp1_to_T.append(torch.zeros(self.B, *shape, device=self.device))

    def loadDataToBatch(self, load_idx: int):
        """
        load data to the batch
        :param load_idx: The index of the batch to load data to
        :return:
        """
        self.dataset.resetSampleLength(random.choice(list(range(64, 513))))
        self.dataset.resetEraseRate(random.uniform(0.2, 0.9))

        # Get the data
        traj_0, erase_mask, lnglat_guess, loc_mean, meta = self.dataset[self.dataset_idx_mapping[self.data_idx]]
        erase_mask = erase_mask.reshape(1, 1, -1)
        lnglat_guess = lnglat_guess.unsqueeze(0)

        # Update the data index
        self.data_idx = (self.data_idx + 1) % len(self.dataset)
        # If we have gone through the entire dataset, shuffle it
        # The next time we will start from the beginning
        if self.data_idx == 0:
            self.dataset_idx_mapping = np.random.permutation(len(self.dataset))

        trajs, mask, comb_noises = self.addNoise(traj_0.unsqueeze(0), erase_mask)
        # trajs: (T+1, 1, 3, L)
        # mask: (1, 3, L)
        # comb_noises: (T, 1, 2, l)

        trajs = trajs[:, 0, ...]  # (T+1, 3, L)
        self.inputs[load_idx] = torch.cat(
            [trajs, lnglat_guess.repeat(self.T + 1, 1, 1), erase_mask.repeat(self.T + 1, 1, 1)], dim=1)  # (T+1, 6, L)
        self.masks[load_idx] = mask[0]
        self.loc_means[load_idx] = loc_mean
        self.metas[load_idx] = meta
        self.eps_0_to_tp1[load_idx] = comb_noises[:, 0, ...]

        # Update tau and tau_next
        self.tau[load_idx] = self.tau_list[1]
        self.tau_next[load_idx] = self.tau_list[0]
        for i in range(len(self.s_tp1_to_T)):
            self.s_tp1_to_T[i][load_idx] = torch.zeros_like(self.s_tp1_to_T[i][load_idx])

    def updateBatch(self):
        """
        Update the batch
        :return: If
        """

        # Update t and t+1
        self.tau = func.relu(self.tau - self.skip_step)
        self.tau_next -= self.skip_step

        min_tau_next_ids = torch.argwhere(self.tau_next <= 0)

        for min_tau_next_id in min_tau_next_ids:
            self.loadDataToBatch(min_tau_next_id)

    def __len__(self):
        return self.total_iterations

    def __iter__(self):
        # first batch_size iterations, we only load data
        for b in range(self.B):
            self.loadDataToBatch(b)
            if self.mode == "Uniform":
                self.tau[b] = max(b * self.skip_step % self.T - 1, 0)
                self.tau_next[b] = ((b + 1) * self.skip_step - 1) % self.T
            elif self.mode == "Consecutive":
                self.tau = func.relu(self.tau - self.skip_step)
                self.tau_next -= self.skip_step
        rprint(f"[green]Batch Fill Complete, Training Starts[/green]")
        print("t = ")
        print(self.tau)
        print("t_next = ")
        print(self.tau_next)

        # Main loop
        for _ in range(self.total_iterations):
            # return tau, tau_next, x_tau, x_tau_next, eps_0_to_tau, eps_0_to_tau_next, state
            yield (self.tau,
                   self.tau_next,
                   self.inputs[self.batch_idx, self.tau + 1],
                   self.inputs[self.batch_idx, self.tau_next + 1],
                   self.inputs[:, -1, ...],
                   [self.eps_0_to_tp1[i][self.tau[i]] for i in range(self.B)],
                   [self.eps_0_to_tp1[i][self.tau_next[i]] for i in range(self.B)],
                   self.masks,
                   self.loc_means,
                   self.metas,
                   self.s_tp1_to_T)

            # The method will be paused here, and resumed when the for loop is called again
            # At the beginning of the loop, we update the batch
            self.updateBatch()

    def updateState(self, s_t_to_T: List[torch.Tensor]):
        """
        Update the state
        :param s_t_to_T: The state from t to T
        :return:
        """
        for i in range(len(s_t_to_T)):
            self.s_tp1_to_T[i] = s_t_to_T[i].detach()


    # @torch.compile
    def addNoise(self, traj_0: torch.Tensor, erase_mask: torch.Tensor):
        """
        :param traj_0: (B, 3, L) lng, lat, time
        :param erase_mask: (B, 1, L) 1 for erased, 0 for not erased
        """

        B, C, L = traj_0.shape

        mask = erase_mask.repeat(1, 3, 1) > 0
        erased_subtraj = traj_0[mask].view(B, 3, -1)  # (B, 3, L_erased)

        mask[:, 2] = False

        interp_lnglat_0 = erased_subtraj[:, :2]

        # step_noises[t] = eps_t_to_tp1
        step_noises = torch.randn(self.T, *interp_lnglat_0.shape).to(interp_lnglat_0.device)
        # comb_noises[t] = eps_0_to_tp1
        comb_noises = step_noises.clone()

        trajs = torch.zeros(self.T + 1, *traj_0.shape).to(traj_0.device)
        trajs[0] = traj_0

        interp_lnglat_t = interp_lnglat_0
        for t in range(1, self.T):
            # eps_0_to_tp1 <- eps_0_to_t, eps_t_to_tp1, t
            comb_noises[t] = self.ddm.combineNoise(comb_noises[t - 1], step_noises[t], t)
            # x_t <- x_0, t,
            interp_lnglat_t = self.ddm.diffusionForwardStep(interp_lnglat_t, t - 1, step_noises[t - 1])
            traj = traj_0.clone()
            traj[mask] = interp_lnglat_t.reshape(-1)
            trajs[t] = traj

        interp_lnglat_t = self.ddm.diffusionForward(interp_lnglat_0, self.T - 1, comb_noises[self.T - 1])
        traj = traj_0.clone()
        traj[mask] = interp_lnglat_t.reshape(-1)
        trajs[self.T] = traj

        return trajs, mask, comb_noises


