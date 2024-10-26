from Models.Basics import *
from Models.StateProp import GRU_Conv
from math import sqrt, ceil


class ResBlock(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 embed_c: int,
                 expand: int):
        super().__init__()

        self.mid_c = in_c * expand

        self.s1 = nn.Sequential(
            nn.GroupNorm(32, in_c + embed_c),
            Swish(),
            nn.Conv1d(in_c + embed_c, self.mid_c, 3, 1, 1),
        )

        self.channel_attn_proj = nn.Sequential(
            nn.Conv1d(embed_c, self.mid_c, 1, 1, 0),
            Swish(),
            nn.Conv1d(self.mid_c, self.mid_c, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.s2 = nn.Sequential(
            nn.GroupNorm(32, self.mid_c),
            Swish(),
            nn.Conv1d(self.mid_c, out_c, 3, 1, 1),
        )

        self.shortcut = nn.Conv1d(in_c, out_c, 1, 1, 0) if in_c != out_c else nn.Identity()

        nn.init.zeros_(self.s2[-1].weight)

    def forward(self, x: torch.Tensor, mix_embed: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        # mix_embed: (B, C_m, 1)
        attn = self.channel_attn_proj(mix_embed)  # (B, mid_c, 1)
        residual = self.s2(self.s1(torch.cat([x, mix_embed.repeat(1, 1, x.shape[2])], dim=1)) * attn)
        return self.shortcut(x) + residual


class AttntionBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, embed_c: int, expend: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.H = expend * 2
        self.d_qk = in_c // 2 * self.H

        self.qkv_proj = nn.Sequential(
            nn.GroupNorm(32, in_c + embed_c),
            Swish(),
            nn.Conv1d(in_c + embed_c, in_c * self.H * 2, 1, 1, 0),
        )

        self.scale = 1 / sqrt(in_c)

        self.out_proj = nn.Sequential(
            nn.GroupNorm(32, in_c * self.H),
            nn.Dropout(dropout),
            nn.Conv1d(in_c * self.H, out_c, 3, 1, 1)
        )

        nn.init.zeros_(self.out_proj[-1].weight)

        self.shortcut = nn.Conv1d(in_c, out_c, 1, 1, 0) if in_c != in_c else nn.Identity()

    def forward(self, x: torch.Tensor, mix_embed: torch.Tensor) -> torch.Tensor:
        # x: (B, C_x, L_x)
        # mix_embed: (B, C_m, 1)
        L_x = x.shape[2]

        qkv = self.qkv_proj(torch.cat([x, mix_embed.repeat(1, 1, L_x)], dim=1))
        q = rearrange(qkv[:, :self.d_qk], 'b (h c) l -> (b h) c l', h=self.H)  # (B*H, C, L)
        k = rearrange(qkv[:, self.d_qk:2*self.d_qk], 'b (h c) l -> (b h) l c', h=self.H)  # (B*H, L, C)
        v = rearrange(qkv[:, 2*self.d_qk:], 'b (h c) l -> (b h) c l', h=self.H)  # (B*H, C, L)

        attn = torch.softmax(torch.bmm(k, q) * self.scale, dim=1)  # (B*H, L, L)

        return self.shortcut(x) + self.out_proj(rearrange(torch.bmm(v, attn), '(b h) c l -> b (h c) l', h=self.H))


class UNetStage(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 embed_c: int,
                 construction_code: str,
                 expend: int,
                 downcale: bool = False,
                 upsample: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__()

        layers = []

        if construction_code[0] == 'A':
            layers.append(AttntionBlock(in_c, out_c, embed_c, expend, dropout))
        elif construction_code[0] == 'R':
            layers.append(ResBlock(in_c, out_c, embed_c, expend))

        for code in construction_code[1:]:
            if code == 'A':
                layers.append(AttntionBlock(out_c, out_c, embed_c, expend, dropout))
            elif code == 'R':
                layers.append(ResBlock(out_c, out_c, embed_c, expend))
            else:
                raise ValueError(f"Invalid code: {code}")

        self.layers = nn.ModuleList(layers)

        if downcale:
            self.out_proj = nn.Conv1d(out_c, out_c, 3, 2, 1)
        elif upsample:
            self.out_proj = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(out_c, out_c, 3, 1, 1)
            )
        else:
            self.out_proj = nn.Conv1d(out_c, out_c, 3, 1, 1)

    def forward(self, x: torch.Tensor, mix_embed: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mix_embed)
        return self.out_proj(x)


class TrajWeaverUNet(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 diffusion_steps: int,
                 c_list: List[int],
                 blocks: List[str],
                 embed_c: int,
                 expend: int,
                 dropout: float = 0.0,
                 ) -> None:
        """
        :param in_c:                * number of channels of the input trajectory representation
        :param diffusion_steps:     * total diffusion steps
        :param c_list:    * channel schedule of the UNet, example: [32, 64, 128, 256]
        :param blocks:        * number of residual blocks in each stage, example: [2, 2, 1]
        :param embedings_c:         * number of channels of the embedding vector
        :param embed_c:        * number of channels of the time embedding vector
        :param traj_context_c:      * number of channels of the broken trajectory context
        :param expend:           * number of heads in the multi-head attention
        :param dropout:             * dropout rate
        :param max_length:          * max length of the trajectory
        :param self_attn:           * Whether to use MHSA as main module
        :param padded:              * Is the length padded
        """
        super().__init__()

        self.c_list = c_list
        self.stages = len(c_list) - 1
        self.embed_c = embed_c

        # This block adds positional encoding and trajectory length encoding to the input trajectory
        self.pre_embed = nn.Conv1d(in_c, c_list[0], 3, 1, 1)

        # This obtains the time embedding of the input trajectory (but not yet added)
        self.time_embedder = NumberEmbedder(max_num=diffusion_steps, hidden_dim=256, embed_dim=embed_c)
        self.length_embedder = nn.Sequential(
            nn.Embedding(513, 256),
            nn.Linear(256, embed_c),
        )

        # Create Encoder (Down sampling) Blocks for UNet
        in_channels = c_list[:-1]
        out_channels = c_list[1:]
        self.down_blocks = nn.ModuleList()
        for i in range(self.stages):
            self.down_blocks.append(
                UNetStage(in_channels[i], out_channels[i], embed_c, blocks[i], expend, downcale=True))

        # Create Middle Attention Block for UNet
        self.mid_attn_block = UNetStage(c_list[-1], c_list[-1], embed_c, "AA", expend)

        # Create Decoder (Up sampling) Blocks for UNet
        self.up_blocks = nn.ModuleList()
        # reverse the channel schedule
        in_channels = c_list[-1:0:-1]
        out_channels = c_list[-2::-1]
        for i in range(self.stages):
            self.up_blocks.append(
                UNetStage(in_channels[i] * 2, out_channels[i], embed_c, blocks[-i], expend, upsample=True))

        self.head = nn.Conv1d(c_list[0], out_c, 3, 1, 1)


    def forward(self,
                x: torch.Tensor,
                diffusion_t: torch.Tensor,
                s_list: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        """
        :param x: (B, 6, L) traj, traj_guess, erase_mask
        :param diffusion_t: (B) the diffusion step
        :param traj_erase: (B, 3, l) the erased trajectory
        :param insertion_mask: (B, l-1) the insertion mask
        :param extra_embed: (B, E) the vector of embeddings
        """

        # Embeddings
        lengths = torch.sum(x[:, 5] > -0.5, dim=1)
        mix_embed = (self.time_embedder(diffusion_t) + self.length_embedder(lengths)).unsqueeze(-1)
        x = self.pre_embed(x)
        s_id = 0

        # Encoder
        x = x + s_list[s_id]
        s_id += 1
        down_states = [x]
        for di, down_stage in enumerate(self.down_blocks):
            x = down_stage(x, mix_embed)
            x = x + s_list[s_id]  # (B, C, L) -> (B, C', L//2)
            s_id += 1
            down_states.append(x)

        # Middle Attention Block
        x = self.mid_attn_block(x, mix_embed)  # (B, C', L//2**i)

        # Decoder
        x = x + s_list[s_id]
        s_id += 1
        up_states = [x]
        for i, up_stage in enumerate(self.up_blocks):
            # fuse with skip connection
            x = torch.cat([x, down_states[-i - 1]], dim=1)  # (B, C*2, L//2**i)
            x = up_stage(x, mix_embed)  # (B, C', L//2**i)
            x = x + s_list[s_id]
            s_id += 1
            up_states.append(x)

        return self.head(up_states[-1]), down_states + up_states

    def getStateShapes(self, traj_len):
        shapes = [(self.c_list[i], ceil(traj_len / (2 ** i))) for i in range(len(self.c_list))]
        return shapes + shapes[::-1]

    def getFeatureShapes(self, traj_len):
        return self.getStateShapes(traj_len)


class Linkage(nn.Module):
    def __init__(self, in_shapes: List[Tuple[int, int]], max_t: int):
        super().__init__()
        self.gru_cells = nn.ModuleList()
        for shape in in_shapes:
            c = shape[0]
            self.gru_cells.append(GRU_Conv(c, c, max_t))

    def forward(self, hidden, s_tp1, t):
        for i in range(len(self.gru_cells)):
            hidden[i] = self.gru_cells[i](hidden[i], s_tp1[i], t)
        return hidden


if __name__ == "__main__":
    model = TrajWeaverUNet(
        in_c=6,  # input trajectory encoding channels
        out_c=2,
        diffusion_steps=500,  # maximum diffusion steps
        c_list=[128, 128, 128, 256],  # channel schedule of stages, first element is stem output channels
        blocks=["RRRR", "RRRR", "RRRR"],  # number of resblocks in each stage
        embed_c=64,  # channels of mix embeddings
        expend=4,  # number of heads for attention
        dropout=0.0,  # dropout
    ).cuda()

    B = 2
    L = 512

    x = torch.randn(B, 6, L, device="cuda")
    prev_features = [torch.zeros(B, *shape, device="cuda") for shape in model.getStateShapes(L)]
    diffusion_t = torch.randint(0, 100, (B,), device="cuda")
    eps, features = model(x, diffusion_t, prev_features)

    for f in features:
        print(f.shape)

    # length: 16 ~ 64
    torch.save(model.state_dict(), 'TrajWeaver7.pth')
    torch.save(Linkage(model.getStateShapes(512), 500).state_dict(), "linkage.pth")