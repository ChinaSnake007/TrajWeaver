from Models.Basics import *


class GRU_Conv(nn.Module):
    """
    This is actually a variant of GRU, but the state is in shape (B, state_c, L),
    The linear operations are replaced with 1D convolutions.
    """
    def __init__(self, in_c: int, state_c: int, max_t: int):
        super().__init__()

        self.t_embedder = nn.Sequential(
            nn.Embedding(max_t, 128),
            nn.Unflatten(-1, (-1, 1))
        )
        self.embed_1 = nn.Conv1d(128, state_c * 3, 1, 1, 0)
        self.embed_2 = nn.Conv1d(128, state_c * 2, 1, 1, 0)

        self.input_proj = nn.Conv1d(in_c, state_c * 3, 3, 1, 1)

        self.state_proj = nn.Conv1d(state_c, state_c * 2, 3, 1, 1)

        self.state_r_proj = nn.Conv1d(state_c, state_c, 3, 1, 1)


    def forward(self, x: Tensor, prev_h: Tensor, t: Tensor):
        # input_state: (B, state_c, L)
        # prev_hidden_state: (B, state_c, L)
        # output: (B, state_c, L) the next hidden state

        t_embed = self.t_embedder(t)

        Wz_xt, Wr_xt, Wh_xt = torch.chunk(self.input_proj(x) + self.embed_1(t_embed), 3, dim=1)

        Uz_ht, Ur_ht = torch.chunk(self.state_proj(prev_h) + self.embed_2(t_embed), 2, dim=1)

        z = torch.sigmoid(Wz_xt + Uz_ht)

        r = torch.sigmoid(Wr_xt + Ur_ht)

        h_tilde = torch.tanh(Wh_xt + self.state_r_proj(r * prev_h))

        h = (1 - z) * prev_h + z * h_tilde

        return h