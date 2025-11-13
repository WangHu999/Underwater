import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ECA attention module
class Attention_eca(nn.Module):
    def __init__(self, num_heads, k_size, bias):
        super(Attention_eca, self).__init__()
        self.num_heads = num_heads

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        heads = x.chunk(self.num_heads, dim=1)
        outputs = []
        for head in heads:
            y = self.avg_pool(head)
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            out = head * y.expand_as(head)
            outputs.append(out)
        # Two different branches of ECA module
        output = torch.cat(outputs, dim=1)

        return output


##########################################################################
class TransformerBlock_eca(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_eca, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_eca(num_heads, 3, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


def function_delta(input):
    out = torch.clamp(input, min=0, max=60)
    return out


class HSV2RGB(nn.Module):
    def __init__(self):
        super(HSV2RGB, self).__init__()

    def forward(self, hsv):
        batch, c, w, height = hsv.size()
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        htemp = (h * 360) % 360
        h = htemp / 360
        # h = h / 360
        vs = torch.div(torch.mul(v, s), 60)  # (v * s) / 60
        R1_delta = function_delta(torch.add(torch.mul(h, 360), -60))  # delta(360h - 60)
        R2_delta = function_delta(torch.add(torch.mul(h, 360), -240))

        G1_delta = function_delta(torch.add(torch.mul(h, 360), 0))
        G2_delta = function_delta(torch.add(torch.mul(h, 360), -180))

        B1_delta = function_delta(torch.add(torch.mul(h, 360), -120))
        B2_delta = function_delta(torch.add(torch.mul(h, 360), -300))

        one_minus_s = torch.mul(torch.add(s, -1), -1)
        R_1 = torch.add(v, torch.mul(vs, R1_delta), alpha=-1)
        R_2 = torch.mul(vs, R2_delta)
        R = torch.add(R_1, R_2)

        G_1 = torch.add(torch.mul(v, one_minus_s), torch.mul(vs, G1_delta))
        G_2 = torch.mul(vs, G2_delta)
        G = torch.add(G_1, G_2, alpha=-1)

        B_1 = torch.add(torch.mul(v, one_minus_s), torch.mul(vs, B1_delta))
        B_2 = torch.mul(vs, B2_delta)
        B = torch.add(B_1, B_2, alpha=-1)

        del h, s, v, vs, R1_delta, R2_delta, G1_delta, G2_delta, B1_delta, B2_delta, one_minus_s, R_1, R_2, G_1, G_2, B_1, B_2

        R = torch.reshape(R, (batch, 1, w, height))
        G = torch.reshape(G, (batch, 1, w, height))
        B = torch.reshape(B, (batch, 1, w, height))
        RGB_img = torch.cat([R, G, B], 1)

        return RGB_img


class RGB2HSV(nn.Module):
    def __init__(self):
        super(RGB2HSV, self).__init__()

    def forward(self, rgb):
        batch, c, w, h = rgb.size()
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        V, max_index = torch.max(rgb, dim=1)
        min_rgb = torch.min(rgb, dim=1)[0]
        v_plus_min = V - min_rgb
        S = v_plus_min / (V + 0.0001)
        H = torch.zeros_like(rgb[:, 0, :, :])
        # if rgb.type() == 'torch.cuda.FloatTensor':
        #     H = torch.zeros(batch, w, h).type(torch.cuda.FloatTensor)
        # else:
        #     H = torch.zeros(batch, w, h).type(torch.FloatTensor)
        mark = max_index == 0
        H[mark] = 60 * (g[mark] - b[mark]) / (v_plus_min[mark] + 0.0001)
        mark = max_index == 1
        H[mark] = 120 + 60 * (b[mark] - r[mark]) / (v_plus_min[mark] + 0.0001)
        mark = max_index == 2
        H[mark] = 240 + 60 * (r[mark] - g[mark]) / (v_plus_min[mark] + 0.0001)

        mark = H < 0
        H[mark] += 360
        H = H % 360
        H = H / 360
        HSV_img = torch.cat([H.view(batch, 1, w, h), S.view(batch, 1, w, h), V.view(batch, 1, w, h)], 1)
        return HSV_img


class LITM(nn.Module):
    def __init__(self):
        super(LITM, self).__init__()
        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        self.rgb2hsv = RGB2HSV()
        self.hsv2rgb = HSV2RGB()

        self.transform1 = TransformerBlock_eca(dim=int(64), num_heads=2, ffn_expansion_factor=2.66,
                                               bias=False, LayerNorm_type='WithBias')

        self.transform2 = TransformerBlock_eca(dim=int(64), num_heads=2, ffn_expansion_factor=2.66,
                                               bias=False, LayerNorm_type='WithBias')

        # rgb
        self.norm_batch = nn.InstanceNorm2d  # choose one

        self.rgb_norm_batch1 = self.norm_batch(64)
        self.rgb_con1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # self.rgb_norm_batch2 = self.norm_batch(64)
        # self.rgb_con2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.rgb_norm_batch3 = self.norm_batch(64)
        # self.rgb_con3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.rgb_norm_batch4 = self.norm_batch(64)
        # self.rgb_con4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.rgb_norm_batch5 = self.norm_batch(64)
        # self.rgb_con5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.rgb_norm_batch6 = self.norm_batch(64)
        # self.rgb_con6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.rgb_con7 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        self.rgb_fuction_down = nn.LeakyReLU(inplace=True)
        self.rgb_fuction_up = nn.ReLU(inplace=True)

        # hsv
        self.relu = nn.LeakyReLU(inplace=True)
        self.rrelu = nn.ReLU(inplace=True)
        self.M = 11
        # New /1/./2/./3/ use number_f = 32
        number_f = 64
        self.e_conv1 = nn.Conv2d(6, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_convfc = nn.Linear(number_f, 44)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.avagepool = nn.AdaptiveAvgPool2d(1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # confidence
        self.norm_batch = nn.InstanceNorm2d  # choose one

        self.norm_batch1 = self.norm_batch(64)
        self.con1 = nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch2 = self.norm_batch(64)
        self.con2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch3 = self.norm_batch(64)
        self.con3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch4 = self.norm_batch(64)
        self.con4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch5 = self.norm_batch(64)
        self.con5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.norm_batch6 = self.norm_batch(64)
        self.con6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.con7 = nn.Conv2d(64, 6, kernel_size=1, stride=1, padding=0)

        self.fuction_down = nn.LeakyReLU(inplace=True)
        self.fuction_up = nn.ReLU(inplace=True)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0, 0.02)
                # m.bias.data.zero_()
                # nn.init.xavier_normal_(m.weight.data)
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
                # nn.init.xavier_normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.rgb_fuction_down(self.rgb_norm_batch1(self.rgb_con1(x)))
        h = self.transform1(h)
        h = self.transform1(h)
        h = self.transform1(h)
        h = self.transform1(h)
        h = self.transform1(h)
        h = self.transform1(h)
        # h = self.rgb_fuction_down(self.rgb_norm_batch2(self.rgb_con2(h)))
        # h = self.rgb_fuction_down(self.rgb_norm_batch3(self.rgb_con3(h)))
        # h = self.rgb_fuction_down(self.rgb_norm_batch4(self.rgb_con4(h)))
        # h = self.rgb_fuction_up(self.rgb_norm_batch5(self.rgb_con5(h)))
        # h = self.rgb_fuction_up(self.rgb_norm_batch6(self.rgb_con6(h)))  # try to use
        rgb_out = torch.sigmoid(self.rgb_con7(h))
        rgb_out = rgb_out[:, 0:3, :, :]
        hsv_fromrgbout = self.rgb2hsv(rgb_out)
        hsv_frominput = self.rgb2hsv(x)

        hsv_input = torch.cat([hsv_fromrgbout, hsv_fromrgbout], dim=1)  # [B, 6, 256, 256]
        batch_size = hsv_input.size()[0]
        x1 = self.relu(self.e_conv1(hsv_input))  # 输入 [B, 6, H, W] → 输出 [B, 64, 256, 256]
        x1 = self.maxpool(x1)                    # 输出 [B, 64, 128, 128]

        # x2 = self.relu(self.e_conv2(x1))          # 输出 [B, 64, 128, 128]
        # x2 = self.maxpool(x2)
        # x3 = self.relu(self.e_conv3(x2))          # 输出 [B, 64, 64, 64]
        # x3 = self.maxpool(x3)
        # x4 = self.relu(self.e_conv4(x3))          # 输出 [B, 64, 32, 32]
        # x_r = self.relu(self.e_conv7(x4))

        x2 = self.transform2(x1)                 # [B, 64, 128, 128]
        x2 = self.maxpool(x2)                    # [B, 64, 64, 64]
        x3 = self.transform2(x2)                 # [B, 64, 64, 64]
        x3 = self.maxpool(x3)                    # [B, 64, 32, 32]
        x4 = self.transform2(x3)                 # [B, 64, 32, 32]
        x_r = self.transform2(x4)

        # self.avagepool(x_r).shape  [2, 64, 1, 1]
        x_r = self.avagepool(x_r).view(batch_size, -1)     # 	[2, 64]

        x_r = self.e_convfc(x_r)                           # 	[B, 44]
        H, S, V, H2S = torch.split(x_r, self.M, dim=1)  # 尺寸大小[B, 64]
        H_in, S_in, V_in = hsv_input[:, 0:1, :, :], hsv_input[:, 1:2, :, :], hsv_input[:, 2:3, :, :],
        H_out = piece_function_org(H_in, H, self.M)  # 对 H_in, S_in, V_in 分别使用 piecewise 映射函数进行非线性增强
        S_out1 = piece_function_org(S_in, S, self.M)
        V_out = piece_function_org(V_in, V, self.M)

        S_out2 = piece_function_org(H_in, H2S, self.M)
        S_out = (S_out1 + S_out2) / 2



        # 对 S_out 和 V_out（即增强后的饱和度和亮度图像）进行数值裁剪（clamping）
        zero_lab = torch.zeros(S_out.shape).cuda()
        s_t = torch.where(S_out < 0, zero_lab, S_out)
        one_lab = torch.ones(S_out.shape).cuda()
        S_out = torch.where(s_t > 1, one_lab, s_t)

        zero_lab = torch.zeros(V_out.shape).cuda()
        s_t = torch.where(V_out < 0, zero_lab, V_out)
        one_lab = torch.ones(V_out.shape).cuda()
        V_out = torch.where(s_t > 1, one_lab, s_t)



        hsv_out = torch.cat([H_out, S_out, V_out], dim=1)
        curve = torch.cat([H.view(batch_size, 1, -1),
                           S.view(batch_size, 1, -1),
                           V.view(batch_size, 1, -1),
                           H2S.view(batch_size, 1, -1)], dim=1)

        hsv_out_rgb = self.hsv2rgb(hsv_out)

        confindencenet_input = torch.cat([x,
                                          rgb_out,
                                          hsv_out_rgb], dim=1)

        h = self.fuction_down(self.norm_batch1(self.con1(confindencenet_input)))


        h = self.fuction_down(self.norm_batch2(self.con2(h)))
        h = self.fuction_down(self.norm_batch3(self.con3(h)))
        h = self.fuction_down(self.norm_batch4(self.con4(h)))
        h = self.fuction_down(self.norm_batch5(self.con5(h)))
        h = self.fuction_down(self.norm_batch6(self.con6(h)))  # try to use

        # h = self.transform1(h)
        # h = self.transform1(h)
        # h = self.transform1(h)
        # h = self.transform1(h)
        # h = self.transform1(h)

        confindence_out = torch.sigmoid(self.con7(h))

        # 需要改名
        confindence_rgb = confindence_out[:, 0:3, :, :]
        confindence_hsv = confindence_out[:, 3:6, :, :]
        output_useconf = 0.5 * confindence_rgb * rgb_out + \
                         0.5 * confindence_hsv * hsv_out_rgb

        return output_useconf
        #, rgb_out, hsv_out_rgb


def piece_function_org(x_m, para_m, M):
    b, c, w, h = x_m.shape
    r_m = para_m[:, 0].view(b, c, 1, 1).expand(b, c, w, h)
    for i in range(M - 1):
        para = (para_m[:, i + 1] - para_m[:, i]).view(b, c, 1, 1).expand(b, c, w, h)
        r_m = r_m + para * \
              sgn_m(M * x_m - i * torch.ones(x_m.shape).cuda())
    return r_m


def sgn_m(x):
    # x = torch.Tensor(x)
    zero_lab = torch.zeros(x.shape).cuda()
    # print("one_lab",one_lab)
    s_t = torch.where(x < 0, zero_lab, x)
    one_lab = torch.ones(x.shape).cuda()
    s = torch.where(s_t > 1, one_lab, s_t)
    return s


if __name__ == "__main__":
    model = LITM().cuda()
    x = torch.ones([2, 3, 256, 256]).cuda()
    y = model(x)
    print(y.shape)
    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params / (1024 * 1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')
