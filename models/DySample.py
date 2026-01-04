import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(module,
                mean=0,
                std=1,
                bias=0):
	if hasattr(module, 'weight') and module.weight is not None:
		nn.init.normal_(module.weight, mean, std)
	if hasattr(module, 'bias') and module.bias is not None:
		nn.init.constant_(module.bias, bias)


def constant_init(module,
                  val,
                  bias=0):
	if hasattr(module, 'weight') and module.weight is not None:
		nn.init.constant_(module.weight, val)
	if hasattr(module, 'bias') and module.bias is not None:
		nn.init.constant_(module.bias, bias)


# 构建 2D sin-cos 位置编码
def build_2d_positional_encoding(H,
                                 W,
                                 num_pos_feats=32,
                                 device="cpu",
                                 dtype=torch.float32):
	y_embed = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).repeat(1, W)  # [H, W]
	x_embed = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).repeat(H, 1)  # [H, W]

	eps = 1e-6
	y_embed = y_embed / (H + eps)
	x_embed = x_embed / (W + eps)

	dim_t = torch.arange(num_pos_feats, dtype=dtype, device=device)
	dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats) #  [num_pos_feats,1]

	pos_x = x_embed.unsqueeze(-1) / dim_t  # [H, W, C]
	pos_y = y_embed.unsqueeze(-1) / dim_t  # [H, W, C]

	pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
	pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)

	pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1).unsqueeze(0)  # [1, C*2, H, W]
	return pos  # shape: [1, num_pos_feats*4, H, W]


class DySample(nn.Module):
	def __init__(self,
	             in_channels,
	             scale=2,
	             style='lp',
	             groups=4,
	             dyscope=False,
	             with_pos_enc=False,
	             num_pos_feats=32):
		super().__init__()
		self.scale = scale
		self.style = style
		self.groups = groups
		self.with_pos_enc = with_pos_enc
		self.num_pos_feats = num_pos_feats

		assert style in ['lp', 'pl']
		if style == 'pl':
			assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
		assert in_channels >= groups and in_channels % groups == 0


		in_channels_with_pos = in_channels + (num_pos_feats * 4) if with_pos_enc else in_channels

		if style == 'pl':
			in_channels_with_pos = in_channels_with_pos // scale ** 2
			out_channels = 2 * groups
		else:
			out_channels = 2 * groups * scale ** 2

		self.offset = nn.Conv2d(in_channels_with_pos, out_channels, 1)
		normal_init(self.offset, std=0.001)
		if dyscope:
			self.scope = nn.Conv2d(in_channels_with_pos, out_channels, 1, bias=False)
			constant_init(self.scope, val=0.)

		self.register_buffer('init_pos', self._init_pos())

	def _init_pos(self):
		h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
		return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

	def sample(self,
	           x,
	           offset):
		B, _, H, W = offset.shape
		offset = offset.view(B, 2, -1, H, W)
		coords_h = torch.arange(H, dtype=x.dtype, device=x.device) + 0.5
		coords_w = torch.arange(W, dtype=x.dtype, device=x.device) + 0.5

		coords = torch.stack(torch.meshgrid([coords_w,
		                                     coords_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0).to(x.device) # 构建采样坐标网格
		normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
		coords = 2 * (coords + offset) / normalizer - 1
		coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
			B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
		return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
		                     align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

	def forward_lp(self,
	               x):
		B, C, H, W = x.shape

		# 构建位置编码并拼接
		if self.with_pos_enc:
			pos_enc = build_2d_positional_encoding(H, W, num_pos_feats=self.num_pos_feats, device=x.device, dtype=x.dtype)
			x_cat = torch.cat([x, pos_enc.repeat(B, 1, 1, 1)], dim=1)
		else:
			x_cat = x

		if hasattr(self, 'scope'):
			offset = self.offset(x_cat) * self.scope(x_cat).sigmoid() * 0.5 + self.init_pos
		else:
			offset = self.offset(x_cat) * 0.25 + self.init_pos

		return self.sample(x, offset)

	def forward_pl(self,
	               x):
		x_ = F.pixel_shuffle(x, self.scale)
		B, C, H, W = x_.shape

		# 构建位置编码并拼接
		if self.with_pos_enc:
			pos_enc = build_2d_positional_encoding(H, W, num_pos_feats=self.num_pos_feats, device=x.device, dtype=x.dtype)
			x_cat = torch.cat([x_, pos_enc.repeat(B, 1, 1, 1)], dim=1)
		else:
			x_cat = x_

		if hasattr(self, 'scope'):
			offset = F.pixel_unshuffle(self.offset(x_cat) * self.scope(x_cat).sigmoid(), self.scale) * 0.5 + self.init_pos
		else:
			offset = F.pixel_unshuffle(self.offset(x_cat), self.scale) * 0.25 + self.init_pos

		return self.sample(x, offset)

	def forward(self,
	            x):
		if self.style == 'pl':
			return self.forward_pl(x)
		return self.forward_lp(x)


# 测试用例
if __name__ == '__main__':
	block = DySample(8, scale=2, style='lp', groups=4, dyscope=True, with_pos_enc=True, num_pos_feats=32)
	input = torch.rand(1, 8, 256, 256)
	output = block(input)
	print(input.size())
	print(output.size())
