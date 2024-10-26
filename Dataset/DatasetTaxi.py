import torch
import torch.utils.data as data
from typing import *
import random
from rich.status import Status

class TaxiDataset(data.Dataset):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __init__(self, max_len: int, load_path: str):

        with Status(f'Loading {load_path} from disk...'):
            dataset_part = torch.load(load_path)
            self.trajs = [sample[0].to(self.device) for sample in dataset_part]

        self.max_length = max_len
        self.sample_length = max_len
        self.erase_rate = 0.0
        self.shift_rate = 0.0


    def resetSampleLength(self, length: int):
        if length > self.max_length:
            length = self.max_length
        elif length < 1:
            length = 1
        self.sample_length = length


    def resetEraseRate(self, elim_rate: float):
        self.erase_rate = min(1.0, max(0.0, elim_rate))


    def resetShiftRate(self, shift_rate: float):
        self.shift_rate = min(1.0, max(0.0, shift_rate))

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



    def __len__(self):
        return len(self.trajs)


    def __getitem__(self, index: Any) -> Any:
        """
        :param index: The index of the trajectory in dataset_part
        :return: lon_lat: (2, N), attr: (3,), times: (N,)
        """
        # lon_lat: (N, 3), times: (N,)
        traj = self.trajs[index]
        sample_start = random.randint(0, self.max_length - self.sample_length)
        traj = traj[:, sample_start:sample_start + self.sample_length].to(self.device)

        n_remain = self.sample_length - int(self.sample_length * self.erase_rate)
        # select n_remain indices to remain, sorted, and exclude the first and last point
        remain_indices = torch.randperm(self.sample_length - 2)[:n_remain - 2].to(self.device) + 1
        remain_indices = torch.sort(remain_indices)[0]
        # add firsst and the last point
        remain_indices = torch.cat([torch.tensor([0], device=self.device), remain_indices,
                                    torch.tensor([self.sample_length - 1], device=self.device)])

        # binary_mask is 0 means the broken traj, 1 means erased
        mask = torch.ones(self.sample_length, dtype=torch.float32, device=self.device)
        mask[remain_indices] = 0

        loc_guess = self.guessTraj(traj, mask)

        pad_size = self.max_length - self.sample_length
        mask = torch.nn.functional.pad(mask, (0, pad_size), mode="constant", value=-1)
        traj = torch.nn.functional.pad(traj, (0, pad_size))
        loc_guess = torch.nn.functional.pad(loc_guess, (0, pad_size))

        return traj, mask, loc_guess

def collate_fn(batch):
    """
    :param batch: list of (traj, mask, loc_guess)
    :return: traj: (B, 3, L), mask: (B, L), loc_guess: (B, 2, L)
    """
    traj, mask, loc_guess = zip(*batch)
    traj = torch.stack(traj, dim=0)
    mask = torch.stack(mask, dim=0)
    loc_guess = torch.stack(loc_guess, dim=0)

    return traj, mask, loc_guess



if __name__ == "__main__":
    from Configs import dataset_args
    dataset = TaxiDataset(**dataset_args)
    for i in range(10):
        test = dataset[i]
