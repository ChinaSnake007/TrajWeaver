import torch
from torch.utils.data import Dataset
from typing import *
import random

Tensor = torch.Tensor

class DatasetApartments(Dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self, max_len: int, load_path: str, centering: bool = False):
        """
        The dataset collected in apartment areas in last-mile delivery
        :param max_len: The maximum length of the trajectory
        :param load_path: The path to load the dataset
        :param centering: Whether to center the trajectory upon preprocessing, default False
        """
        temp = torch.load(load_path)
        self.centering = centering

        # each traj: (3, L). Normalized (lng, lat, sec_of_day)
        self.trajs: List[Tensor] = [each.to(torch.float32).to(self.device) for each in temp["trajs"]]

        # each metadata: (4, L), integers
        # Number of different categories: 25, 21, 3, 2
        self.metadata: List[Tensor] = [each.to(torch.long).to(self.device) for each in temp[" metadata"]]

        self.max_length = max_len
        self.sample_length = max_len
        self.erase_rate = 0.0

    def resetSampleLength(self, length: int):
        self.sample_length = min(self.max_length, max(1, length))

    def resetEraseRate(self, elim_rate: float):
        self.erase_rate = min(1.0, max(0.0, elim_rate))

    @staticmethod
    # @torch.compile
    def guessTraj(traj_0, erase_mask):
        """
        Obtain the guessed trajectory from the original trajectory and the erase mask
        :param traj_0: (3, L)
        :param mask:  (L,)
        :return: guessed locations: (2, L)
        """
        boolean_mask = erase_mask > 0.1  # (L,)

        erased_subtraj = traj_0[:, boolean_mask]  # (3, L_erased)
        remain_subtraj = traj_0[:, ~boolean_mask]  # (3, L_remain)

        L_remain = remain_subtraj.shape[-1]

        time_interp = erased_subtraj[2]  # (L_erased)
        time_remain = remain_subtraj[2]  # (L_remain)
        ids_right = torch.searchsorted(time_remain, time_interp).to(torch.long)  # (L_erased)
        ids_left = ids_right - 1  # (L_erased)

        ids_left = torch.clamp(ids_left, 0, L_remain - 1)
        ids_right = torch.clamp(ids_right, 0, L_remain - 1)

        traj_left = remain_subtraj[:, ids_left]
        traj_right = remain_subtraj[:, ids_right]

        ratio = (time_interp - traj_left[2]) / (traj_right[2] - traj_left[2])  # (L_erased)

        erased_loc_guess = traj_left[:2] * (1 - ratio) + traj_right[:2] * ratio  # (2, L_erased)

        loc_guess = traj_0[:2].clone()  # (2, L)
        loc_guess[:, boolean_mask] = erased_loc_guess

        nan_mask = torch.isnan(loc_guess)

        loc_guess[nan_mask] = torch.zeros_like(loc_guess[nan_mask])

        return loc_guess


    @staticmethod
    def centerTraj(traj: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Center the trajectory to the origin
        :param traj: The trajectory to center, (B, 3, L) or (3, L)
        :return: The centered trajectory and the center, (B, 3, L), (B, 2, 1) or (3, L), (2, 1)
        """
        if len(traj.shape) == 3:
            center = traj[:, :2].mean(dim=2, keepdim=True)
        else:
            center = traj[:2].mean(dim=1, keepdim=True)
        return traj - center, center

    @staticmethod
    def decenterTraj(traj: Tensor, center: Tensor) -> Tensor:
        """
        Decenter the trajectory
        :param traj: The trajectory to decenter, (3, L) or (B, 3, L)
        :param center: The center, (2, 1) or (B, 2, 1)
        :return: The decentered trajectory, (3, L) or (B, 3, L)
        """
        return traj + center


    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        traj = self.trajs[index]
        if self.centering:
            traj, loc_mean = self.centerTraj(traj)
        else:
            loc_mean = torch.zeros(2, 1, device=self.device)
        sample_start = random.randint(0, max(0, traj.shape[1] - self.sample_length))
        traj = traj[:, sample_start:sample_start + self.sample_length]
        meta = self.metadata[index][:, sample_start:sample_start + self.sample_length]

        actual_sample_length = traj.shape[1]

        n_remain = actual_sample_length - int(actual_sample_length * self.erase_rate)
        # select n_remain indices to remain, sorted, and exclude the first and last point
        remain_indices = torch.randperm(actual_sample_length - 2)[:n_remain - 2].to(self.device) + 1
        remain_indices = torch.sort(remain_indices)[0]
        # add first and the last point
        remain_indices = torch.cat([torch.tensor([0], device=self.device), remain_indices,
                                    torch.tensor([actual_sample_length - 1], device=self.device)])

        # binary_mask is 0 means the broken traj, 1 means erased
        mask = torch.ones(actual_sample_length, dtype=torch.float32, device=self.device)
        mask[remain_indices] = 0

        loc_guess = self.guessTraj(traj, mask)

        pad_size = self.max_length - actual_sample_length
        mask = torch.nn.functional.pad(mask, (0, pad_size), mode="constant", value=-1)
        traj = torch.nn.functional.pad(traj, (0, pad_size))
        loc_guess = torch.nn.functional.pad(loc_guess, (0, pad_size))
        meta = torch.nn.functional.pad(meta + 1, (0, pad_size), mode="constant", value=0)

        return traj, mask, loc_guess, loc_mean, meta


if __name__ == "__main__":
    dataset = DatasetApartments(max_len=512, load_path="Submission/Proj2407/dataset.pth")
    dataset.resetEraseRate(0.5)

    traj, mask, loc_guess, loc_mean, meta = dataset[0]
    print(traj.shape, mask.shape, loc_guess.shape, loc_mean.shape, meta.shape)
