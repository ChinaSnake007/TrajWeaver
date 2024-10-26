# TrajWeaver
from DDM import DDIM
from BatchManagers import ThreadedScheduler

# Utils and Configs
from Utils import MovingAverage, loadModel, saveModel, MaskedMSE
from Configs import *

# Eval & Dataset
from eval import recovery


# torch imports
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# Other imports
from math import isnan
import random
from tqdm import tqdm
import os


def train():

    dataset = DatasetApartments(**dataset_args)
    unet = TrajWeaver(**TW_args).cuda()
    linkage = Linkage(unet.getStateShapes(TRAJ_LEN), **link_args).cuda()
    embedder = Embedder(embed_dim).cuda()

    # loadModel("Runs/2024-07-15_05-26-26/last.pth", unet=unet, linkage=linkage, embedder=embedder)

    unet.train()
    linkage.train()
    embedder.train()

    # --- Prepare ---
    diff_manager = DDIM(**diffusion_args)

    loss_func = MaskedMSE()

    # Embedder should have a smaller learning rate
    # because it is trained on every sample for many times with the same input
    optimizer = optim.AdamW([
        {"params": unet.parameters()},
        {"params": linkage.parameters()},
        {"params": embedder.parameters(), "lr": init_lr * 0.01}
    ], lr=init_lr)

    mov_avg_loss = MovingAverage(mov_avg_interval)
    os.makedirs(save_dir)
    writer = SummaryWriter(log_dir)

    with open(log_dir + "info.txt", "w") as file:
        file.write(f"Training {unet.__class__.__name__} on 20240711 dataset\n")
        file.write("Model:\n")
        file.write(str(unet))

    batch_manager = BatchManager(
        ddm=diff_manager,
        skip_step=diffusion_args["skip_step"],
        device="cuda",
        num_epochs=epochs,
        batch_size=batch_size,
        traj_len=TRAJ_LEN,
        dataset=dataset
    )

    batch_manager.dataset.resetSampleLength(random.choice(list(range(64, 513))))
    batch_manager.dataset.resetEraseRate(random.uniform(0.2, 0.9))

    # Register states
    for shape in unet.getStateShapes(TRAJ_LEN):
        batch_manager.registerState(shape)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                     factor=lr_reduce_factor,
                                     patience=int(lr_reduce_patience),
                                     min_lr=1e-6,
                                     verbose=True)

    # The proposed training algorithm with 2 denoising steps trained in one iteration
    best_recovery_loss = 1000
    with ThreadedScheduler(batch_manager, 3) as data_iterator:
        pbar = tqdm(data_iterator, desc="Training", ncols=100)
        for global_it, batch_data in enumerate(pbar):
            optimizer.zero_grad()

            # Get data
            t, tp1, x_t, x_tp1, x_T, eps_0_to_t, eps_0_to_tp1, masks, loc_mean, meta, s_tp1 = batch_data

            # Train Embedder
            embed = embedder(meta, loc_mean)
            # The first denoising step
            output_tp1, hidden = unet(torch.cat([x_tp1, embed], dim=1), tp1, s_tp1)
            # Linkage in between
            s_t = linkage(hidden, s_tp1, tp1)
            # The second denoising step
            output_t, _ = unet(torch.cat([x_t, embed], dim=1), t, s_t)

            # Loss for the first denoising step & embedding
            loss_tp1 = loss_func(output_tp1, eps_0_to_tp1, masks)
            # Loss for the second denoising step & embedding & linkage
            loss_t = loss_func(output_t, eps_0_to_t, masks)
            loss = loss_tp1 + loss_t

            loss_float = loss.item()

            if isnan(loss_float):
                pbar.set_postfix_str("Nan Loss detected")
                continue

            # At the beginning, all states are initialize to 0 regardless of the actual diffusion step
            # However, state should only be 0 at t=T-1
            # So we don't back-propagate the loss until the states values are updated
            if global_it > max(actual_diff_step, batch_size):
                loss.backward()
                optimizer.step()

            batch_manager.updateState(s_t)

            mov_avg_loss << loss_float

            if global_it % 10 == 0:
                pbar.set_postfix_str(f"loss={float(mov_avg_loss):.7f} | lr={optimizer.param_groups[0]['lr']:.4e}")

            if global_it % log_interval == 0:
                writer.add_scalar("Loss", float(mov_avg_loss), global_it)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_it)

            if global_it % 500 == 0:
                saveModel(save_dir + "last.pth", unet=unet, linkage=linkage, embedder=embedder)
                lr_scheduler.step(float(mov_avg_loss))

            if global_it % 1000 == 0:
                recovery_loss, fig = recovery(diff_manager, unet, linkage, embedder)
                writer.add_scalar("Recovery Loss", recovery_loss, global_it)
                writer.add_figure("Recovery Figure", fig, global_it)

                if recovery_loss < best_recovery_loss:
                    best_recovery_loss = recovery_loss
                    saveModel(save_dir + "best.pth", unet=unet, linkage=linkage, embedder=embedder)

if __name__ == "__main__":
    train()


