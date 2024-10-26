import torch
import matplotlib.pyplot as plt
from Configs import *


def plotBatchLoc(loc: Tensor, is_scatter: bool, color):
    for b in range(loc.shape[0]):
        if is_scatter:
            plt.scatter(loc[b, 0].cpu().numpy(), loc[b, 1].cpu().numpy(), s=1, c=color)
        else:
            plt.plot(loc[b, 0].cpu().numpy(), loc[b, 1].cpu().numpy(), linewidth=1, color=color)


def denormalize(loc: Tensor, token_mean, token_std):
    return (loc * token_std[:, :2, None]) + token_mean[:, :2, None]


def recovery(ddm, unet, linkage, embedder, verbose=False):
    """
    :param unet: TrajWeaver7
    :param loc_0: (B, 2, L)
    :param loc_T: (B, 2, L)
    :param time: (B, 1, L)
    :param loc_guess: (B, 2, L)
    :param mask: (B, 1, L)
    :param query_len: (B, )
    :param observe_len: (B, )
    """
    unet = unet.eval()
    linkage = linkage.eval()
    embedder = embedder.eval()

    batch_data = torch.load("Dataset/test_20240711_B100_l512_E05.pth")

    loc_0, loc_T, loc_guess, loc_mean, meta, time, mask, bool_mask, query_len, observe_len = batch_data

    B = loc_0.shape[0]

    with torch.no_grad():
        E = embedder(meta, loc_mean)

    s_T = []
    for shape in unet.getStateShapes(TRAJ_LEN):
        s_T.append(torch.zeros(B, *shape, dtype=torch.float32, device="cuda"))

    loc_rec = ddm.diffusionBackwardWithE(unet, linkage, E, loc_T, s_T, time, loc_guess, mask)

    loc_0_query_part = loc_0[bool_mask]
    loc_rec_query_part = loc_rec[bool_mask]

    mse = torch.nn.functional.mse_loss(loc_rec_query_part, loc_0_query_part) * 1000

    fig = plt.figure()
    plt.title("Original vs Recovery")
    plotBatchLoc(loc_0 + loc_mean, True, "blue")
    plotBatchLoc(loc_rec + loc_mean, True, "red")

    unet.train()
    linkage.train()
    embedder.train()

    return mse, fig