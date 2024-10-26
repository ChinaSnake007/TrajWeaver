import torch
from typing import *
from tqdm import tqdm
from math import log

Tensor = torch.Tensor


class DDPM:
    def __init__(self,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.002,
                 max_diffusion_step: int = 100,
                 device: str = 'cuda',
                 scale_mode: Literal["linear", "quadratic", "log"] = "linear"):

        if scale_mode == "quadratic":
            betas = torch.linspace(min_beta ** 0.5, max_beta ** 0.5, max_diffusion_step).to(device) ** 2
        elif scale_mode == "log":
            betas = torch.exp(torch.linspace(log(min_beta), log(max_beta), max_diffusion_step).to(device))
        else:
            betas = torch.linspace(min_beta, max_beta, max_diffusion_step).to(device)

        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.T = max_diffusion_step
        self.b = betas.view(-1, 1, 1)  # (T, 1, 1)
        self.a = alphas.view(-1, 1, 1)  # (T, 1, 1)
        self.abar = alpha_bars.view(-1, 1, 1)  # (T, 1, 1)
        self.sqrt_abar = torch.sqrt(alpha_bars).view(-1, 1, 1)  # (T, 1, 1)
        self.sqrt_1_m_abar = torch.sqrt(1 - alpha_bars).view(-1, 1, 1)  # (T, 1, 1)

    def diffusionForward(self, x_0, t, eps_0_to_tp1):
        """
        Forward Diffusion Process
        :param x_0: input (B, C, L)
        :param t: time steps (B, )
        :param epsilon: noise (B, C, L)
        :return: x_t: output (B, C, L)
        """
        x_tp1 = self.sqrt_abar[t] * x_0 + self.sqrt_1_m_abar[t] * eps_0_to_tp1
        return x_tp1

    def diffusionForwardStep(self, x_t, t, eps_t_to_tp1):
        # t: 0 - T-1
        return torch.sqrt(self.a[t]) * x_t + torch.sqrt(1 - self.a[t]) * eps_t_to_tp1

    def diffusionBackwardStep(self, x_tp1: torch.Tensor, t: int, epsilon_pred: torch.Tensor):
        """
        Backward Diffusion Process
        :param x_t: input images (B, C, L)
        :param t: time steps
        :param epsilon_pred: predicted noise (B, C, L)
        :param scaling_factor: scaling factor of noise
        :return: x_t-1: output images (B, C, L)
        """
        beta = self.b[t].view(-1, 1, 1)
        alpha = self.a[t].view(-1, 1, 1)
        sqrt_1_minus_alpha_bar = self.sqrt_1_m_abar[t].view(-1, 1, 1)

        mu = (x_tp1 - beta / sqrt_1_minus_alpha_bar * epsilon_pred) / torch.sqrt(alpha)

        if t <= 5:
            return mu
        else:
            stds = torch.sqrt((1 - self.abar[t - 1]) / (1 - self.abar[t]) * beta) * torch.randn_like(x_tp1) * 0.2
            return mu + stds

    @torch.no_grad()
    def diffusionBackward(self,
                          unet: torch.nn.Module,
                          linkage,
                          loc_T: Tensor,
                          s: Tensor,
                          time, loc_guess, mask, verbose=False):
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
        tensor_t = torch.arange(self.T, dtype=torch.long, device=loc_T.device).repeat(B, 1)  # (B, T)
        pbar = tqdm(range(self.T - 1, -1, -1)) if verbose else range(self.T - 1, -1, -1)
        for t in pbar:
            output, hidden = unet(torch.cat([loc_t, additional], dim=1), tensor_t[:, t], s)  # output: (B, 2, L)
            s = linkage(hidden, s, tensor_t[:, t])

            temp = self.diffusionBackwardStep(loc_t, t, output)
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

        term_1 = torch.sqrt(self.a[t]) * self.sqrt_1_m_abar[t - 1] * eps_0_to_t

        term_2 = torch.sqrt(1 - self.a[t]) * eps_t_to_tp1

        return (term_1 + term_2) / self.sqrt_1_m_abar[t]
