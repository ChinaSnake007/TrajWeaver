# encoding: utf-8

import torch
from typing import *
from tqdm import tqdm
from math import log

Tensor = torch.Tensor


# import cv2

class DDIM:
    def __init__(self,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.002,
                 max_diffusion_step: int = 100,
                 device: str = 'cuda',
                 scale_mode: Literal["linear", "quadratic", "log"] = "linear",
                 skip_step=1):

        if scale_mode == "quadratic":
            betas = torch.linspace(min_beta ** 0.5, max_beta ** 0.5, max_diffusion_step).to(device) ** 2
        elif scale_mode == "log":
            betas = torch.exp(torch.linspace(log(min_beta), log(max_beta), max_diffusion_step).to(device))
        else:
            betas = torch.linspace(min_beta, max_beta, max_diffusion_step).to(device)

        self.skip_step = skip_step

        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.T = max_diffusion_step
        self.β = betas.view(-1, 1, 1)  # (T, 1, 1)
        self.α = alphas.view(-1, 1, 1)  # (T, 1, 1)
        self.αbar = alpha_bars.view(-1, 1, 1)  # (T, 1, 1)
        self.sqrt_αbar = torch.sqrt(alpha_bars).view(-1, 1, 1)  # (T, 1, 1)
        self.sqrt_1_m_αbar = torch.sqrt(1 - alpha_bars).view(-1, 1, 1)  # (T, 1, 1)

        self.σ = 0.0

    def diffusionForwardStep(self, x_t, t, ϵ_t_to_tp1):
        # t: 0 - T-1
        # For DDPM, x_t+1 = √(α_t) * x_t + √(1 - α_t) * ϵ_t:t+1
        # For DDIM, x_t+1 = √(αbar_t-2)
        return torch.sqrt(self.α[t]) * x_t + torch.sqrt(1 - self.α[t]) * ϵ_t_to_tp1

    def diffusionForward(self, x_0, t, ϵ):
        """
        Forward Diffusion Process
        :param x_0: input (B, C, L)
        :param t: time steps (B, )
        :param ϵ: noise (B, C, L)
        :return: x_t: output (B, C, L)
        """
        x_t = self.sqrt_αbar[t] * x_0 + self.sqrt_1_m_αbar[t] * ϵ
        return x_t

    def diffusionBackwardStep(self, x_tp1: Tensor, t: int, next_t: int, ϵ_pred: Tensor):
        """
        Backward Diffusion Process
        :param x_t: input images (B, C, L)
        :param t: time steps
        :param ϵ_pred: predicted noise (B, C, L)
        :param scaling_factor: scaling factor of noise
        :return: x_t-1: output images (B, C, L)
        """
        pred_x0 = (x_tp1 - self.sqrt_1_m_αbar[t] * ϵ_pred) / self.sqrt_αbar[t]
        if t <= self.skip_step:
            return pred_x0
        return self.diffusionForward(pred_x0, next_t, ϵ_pred)

    @torch.no_grad()
    def diffusionBackward(self,
                          unet: torch.nn.Module,
                          linkage,
                          loc_T: Tensor,
                          s: Tensor,
                          time, loc_guess, mask,
                          verbose=False):
        """
        Backward Diffusion Process
        :param unet: UNet
        :param input_T: input images (B, 6, L)
        :param s: initial state (B, 32, L//4)
        :param E: mix context (B, 32, L//4)
        :param mask: mask (B, 1, L), 1 for erased, 0 for not erased, -1 for padding
        :param query_len: query length (B, )
        """
        B = loc_T.shape[0]
        # construct update mask, selecting only the part of the input that needs to be updated
        mask_2d = (mask > 0.1).repeat(1, 2, 1)
        additional = torch.cat([time, loc_guess, mask], dim=1)
        loc_t = loc_T.clone()
        tensor_t = torch.arange(self.T, dtype=torch.long, device=loc_t.device).repeat(B, 1)  # (B, T)
        t_list = list(range(self.T - 1, -1, -self.skip_step))
        if t_list[-1] != 0:
            t_list.append(0)
        pbar = tqdm(t_list, desc="Recovery") if verbose else t_list
        for ti, t in enumerate(pbar):
            output, hidden = unet(torch.cat([loc_t, additional], dim=1), tensor_t[:, t], s)  # output: (B, 2, L)
            s = linkage(hidden, s, tensor_t[:, t])
            t_next = 0 if ti + 1 == len(t_list) else t_list[ti + 1]

            temp = self.diffusionBackwardStep(loc_t, t, t_next, output)
            loc_t[mask_2d] = temp[mask_2d]

        return loc_t

    @torch.no_grad()
    def diffusionBackwardDDM(self,
                             unet: torch.nn.Module,
                             loc_T: Tensor,
                             time, loc_guess, mask,
                             verbose=False):
        """
        Backward Diffusion Process
        :param unet: UNet
        :param input_T: input images (B, 6, L)
        :param s: initial state (B, 32, L//4)
        :param E: mix context (B, 32, L//4)
        :param mask: mask (B, 1, L), 1 for erased, 0 for not erased, -1 for padding
        :param query_len: query length (B, )
        """
        B = loc_T.shape[0]
        # construct update mask, selecting only the part of the input that needs to be updated
        mask_2d = (mask > 0.1).repeat(1, 2, 1)
        additional = torch.cat([time, loc_guess, mask], dim=1)
        loc_t = loc_T.clone()
        tensor_t = torch.arange(self.T, dtype=torch.long, device=loc_t.device).repeat(B, 1)  # (B, T)
        t_list = list(range(self.T - 1, -1, -self.skip_step))
        if t_list[-1] != 0:
            t_list.append(0)
        print(t_list)
        pbar = tqdm(t_list) if verbose else t_list
        for ti, t in enumerate(pbar):
            output = unet(torch.cat([loc_t, additional], dim=1), tensor_t[:, t])  # output: (B, 2, L)
            t_next = 0 if ti + 1 == len(t_list) else t_list[ti + 1]
            temp = self.diffusionBackwardStep(loc_t, t, t_next, output)
            loc_t[mask_2d] = temp[mask_2d]

        return loc_t

    @torch.no_grad()
    def diffusionBackwardWithE(self,
                          unet: torch.nn.Module,
                          linkage,
                          E,
                          loc_T: Tensor,
                          s: Tensor,
                          time, loc_guess, mask,
                          verbose=False):
        """
        Backward Diffusion Process
        :param unet: UNet
        :param input_T: input images (B, 6, L)
        :param s: initial state (B, 32, L//4)
        :param E: mix context (B, 32, L//4)
        :param mask: mask (B, 1, L), 1 for erased, 0 for not erased, -1 for padding
        :param query_len: query length (B, )
        """
        B = loc_T.shape[0]
        # construct update mask, selecting only the part of the input that needs to be updated
        mask_2d = (mask > 0.1).repeat(1, 2, 1)
        additional = torch.cat([time, loc_guess, mask, E], dim=1)
        loc_t = loc_T.clone()
        tensor_t = torch.arange(self.T, dtype=torch.long, device=loc_t.device).repeat(B, 1)  # (B, T)
        t_list = list(range(self.T - 1, -1, -self.skip_step))
        if t_list[-1] != 0:
            t_list.append(0)
        pbar = tqdm(t_list) if verbose else t_list
        for ti, t in enumerate(pbar):
            output, hidden = unet(torch.cat([loc_t, additional], dim=1), tensor_t[:, t], s)  # output: (B, 2, L)
            s = linkage(hidden, s, tensor_t[:, t])
            t_next = 0 if ti + 1 == len(t_list) else t_list[ti + 1]

            temp = self.diffusionBackwardStep(loc_t, t, t_next, output)
            loc_t[mask_2d] = temp[mask_2d]

        return loc_t


    def combineNoise(self, eps_0_to_t, eps_t_to_tp1, t):
        """

        :param eps_0_to_t: Combined noise,  (B, 2, L)
        :param eps_t_to_tp1: Noise for step, (B, 2, L)
        :param t: t int {0, 1, 2, ... T-1}
        :return: eps_0_to_tp1
        """
        if t == 0:
            return eps_t_to_tp1

        term_1 = torch.sqrt(self.α[t]) * self.sqrt_1_m_αbar[t - 1] * eps_0_to_t

        term_2 = torch.sqrt(1 - self.α[t]) * eps_t_to_tp1

        return (term_1 + term_2) / self.sqrt_1_m_αbar[t]