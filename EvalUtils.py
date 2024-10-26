import numpy as np
import torch
import ctypes
eval_c_lib = ctypes.cdll.LoadLibrary("./NDTW.dll")
eval_c_lib.NDTW.restype = ctypes.c_float

Tensor = torch.Tensor



def JSD(original_data: Tensor, generated_data: Tensor, n_grids: int = 64, normalize: bool = True):
    # traj_dataset: (B, 2, L)
    # Get the distribution of all points in the dataset
    # return: (B, 2, L)

    # (2BL, 2)
    all_points = torch.cat([original_data.transpose(1, 2).reshape(-1, 2),
                            generated_data.transpose(1, 2).reshape(-1, 2)], dim=0)
    # STEP 1. get min and max
    min_lon = all_points[:, 0].min().item()
    max_lon = all_points[:, 0].max().item()
    min_lat = all_points[:, 1].min().item()
    max_lat = all_points[:, 1].max().item()

    # STEP 2. split city into 16x16 grid
    lng_interval = (max_lon - min_lon) / n_grids
    lat_interval = (max_lat - min_lat) / n_grids

    # STEP 3. count points in each grid
    # point_count = torch.zeros((n_grids, n_grids), device=traj_dataset.device)

    result = []

    for data in [original_data, generated_data]:
        lng_indices = torch.clip((data[:, 0] - min_lon) // lng_interval, 0, n_grids - 1).to(torch.long)
        lat_indices = torch.clip((data[:, 1] - min_lat) // lat_interval, 0, n_grids - 1).to(torch.long)

        eval_c_lib.accumulateCount.restype = ctypes.POINTER(ctypes.c_float * n_grids * n_grids)
        lng_indices = lng_indices.cpu().numpy().astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        lat_indices = lat_indices.cpu().numpy().astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        # The C function eval_c_lib.accumulateCount does this:
        # point_count = np.zeros((n_grids, n_grids))
        # for r, c in zip(lng_indices, lat_indices):
        #     point_count[r, c] += 1
        # return point_count
        # Python is insanely slow, so we use C to do this
        point_count_ptr = eval_c_lib.accumulateCount(lng_indices, lat_indices, n_grids, data.shape[0])

        point_count = torch.tensor(np.frombuffer(point_count_ptr.contents, dtype=np.int32).reshape(n_grids, n_grids)).cuda().to(torch.float32)

        # STEP 4. normalize
        if normalize:
            point_count = (point_count + 1) / point_count.sum()
        result.append(point_count)

    P, Q = result

    P_avg = 0.5 * (P + Q)
    kl_divergence_P = torch.nn.functional.kl_div(P.log(), P_avg, reduction='batchmean')

    # Compute KL divergence between Q and the average distribution of P and Q
    Q_avg = 0.5 * (P + Q)
    kl_divergence_Q = torch.nn.functional.kl_div(Q.log(), Q_avg, reduction='batchmean')

    # Compute Jensen-Shannon Divergence
    jsd_score = 0.5 * (kl_divergence_P + kl_divergence_Q)

    return jsd_score.item()


def NDTW(target_traj, compare_traj):
    """
    This function calculates the Dynamic Time Warping (DTW) distance between two trajectories.
    :param target_traj: trajectory 1 (3, N)
    :param compare_traj: trajectory 2 (3, M)
    :return: DTW distance
    """
    n = target_traj.shape[1]
    m = compare_traj.shape[1]
    dtw = torch.zeros((n + 1, m + 1))
    dtw[1:, 0] = torch.inf
    dtw[0, 1:] = torch.inf
    dtw[0, 0] = 0

    lng_lat_A = target_traj[:2, :].unsqueeze(2)
    lng_lat_B = compare_traj[:2, :].unsqueeze(1)
    squared_dist = torch.sum((lng_lat_A - lng_lat_B) ** 2, dim=0)
    dist_mat = torch.sqrt(squared_dist)

    dist_mat_ptr = dist_mat.cpu().numpy().astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    dtw_ptr = dtw.cpu().numpy().astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    return eval_c_lib.NDTW(dist_mat_ptr, dtw_ptr, n, m)