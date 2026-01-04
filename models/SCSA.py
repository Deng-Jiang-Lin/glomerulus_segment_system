import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F




class PatchEmbed2D(nn.Module):
    """
    Image to Patch Embedding
    Args:
        patch_size (int): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        return self.norm(x)


class SCSA(nn.Module):
    def __init__(
            self,
            dim: int = 96,
            in_chn: int = 64,
            patch_size: int = 4,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
    ):
        super().__init__()
        self.dim = dim
        self.group_chans = self.dim
        assert in_chn % 4 == 0, 'dim must be divisible by 4'

        def dwc(k):
            group_chans = self.group_chans // 4
            return nn.Conv2d(group_chans, group_chans, kernel_size=k, padding=k // 2,
                             groups=group_chans)

        self.PatchEmbed2D = PatchEmbed2D(in_chans=in_chn, patch_size=patch_size, embed_dim=self.dim)
        self.local_dwc = dwc(group_kernel_sizes[0])
        self.global_dwc_s = dwc(group_kernel_sizes[1])
        self.global_dwc_m = dwc(group_kernel_sizes[2])
        self.global_dwc_l = dwc(group_kernel_sizes[3])
        self.conv_h = nn.Conv2d(self.dim, self.dim, kernel_size=(1, 3), padding=(0, 3 // 2), groups=self.dim)
        self.conv_w = nn.Conv2d(self.dim, self.dim, kernel_size=(3, 1), padding=(3 // 2, 0), groups=self.dim)
        self.out_conv = nn.Conv2d(self.dim, in_chn, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x = self.PatchEmbed2D(x)  # b x p_num x p_num x emb_dim
        x = x.permute(0, 3, 1, 2)
        x_h = self.conv_h(x)
        x_w = self.conv_w(x)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.chunk(x_h, 4, dim=1)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.chunk(x_w, 4, dim=1)
        x_h = torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l)
        ), dim=1)

        x_w = torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)
        x_h = self.vssblock(x_h.permute(0, 2, 3, 1))
        x_w = self.vssblock(x_w.permute(0, 2, 3, 1))
        out = torch.add(x_h, x_w)
        out = out.permute(0, 3, 1, 2)
        out = self.out_conv(out)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=False)
        return out

#
# if __name__ == '__main__':
#     img = torch.rand(1, 64, 256, 256).cuda()
#     img2 = torch.rand(1, 128, 128, 128).cuda()
#     model = SCSA(in_chn=128, dim=192).cuda()
#     x = model(img2)
#     print(x.shape)
