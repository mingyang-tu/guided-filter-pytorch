import torch
import torch.nn as nn
import torch.nn.functional as F


def diff_x(x, r):
    left = x[..., r : 2 * r + 1, :]
    middle = x[..., 2 * r + 1 :, :] - x[..., : -2 * r - 1, :]
    right = x[..., -1:, :] - x[..., -2 * r - 1 : -r - 1, :]
    output = torch.cat([left, middle, right], dim=-2)
    return output


def diff_y(x, r):
    left = x[..., r : 2 * r + 1]
    middle = x[..., 2 * r + 1 :] - x[..., : -2 * r - 1]
    right = x[..., -1:] - x[..., -2 * r - 1 : -r - 1]
    output = torch.cat([left, middle, right], dim=-1)
    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = r

    def forward(self, x):
        assert x.dim() == 4
        x = F.pad(x, (self.r, self.r, self.r, self.r), mode="reflect")
        x = x.cumsum(dim=-2)
        x = diff_x(x, self.r)
        x = x.cumsum(dim=-1)
        x = diff_y(x, self.r)
        return x[..., self.r : -self.r, self.r : -self.r] / (2 * self.r + 1) ** 2


class ColorGuidedFilter(nn.Module):
    def __init__(self, radius, epsilon=1e-2):
        super().__init__()
        self.radius = radius
        self.epsilon = epsilon
        self.box_filter = BoxFilter(radius)

    def compute_mean(self, x):
        if x.dim() == 4:
            return self.box_filter(x)
        elif x.dim() == 5:
            B, C1, C2, H, W = x.size()
            x_reshaped = x.view(B * C1, C2, H, W)
            mean = self.box_filter(x_reshaped)
            return mean.view(B, C1, C2, H, W)
        else:
            raise ValueError("Unsupported tensor shape for compute_mean")

    def forward(self, I, p):
        """
        I      : (B, C, H, W)。
        p      : (B, 1, H, W)。
        return : (B, 1, H, W)。
        """
        B, C, H, W = I.size()

        mean_I = self.compute_mean(I)  # (B, C, H, W)
        mean_p = self.compute_mean(p)  # (B, 1, H, W)

        mean_Ip = self.compute_mean(I * p)  # (B, C, H, W)
        cov_Ip = mean_Ip - mean_I * mean_p  # (B, C, H, W)

        mean_II = self.compute_mean(I.unsqueeze(2) * I.unsqueeze(1))  # (B, C, C, H, W)

        cov_II = mean_II - mean_I.unsqueeze(2) * mean_I.unsqueeze(1)  # (B, C, C, H, W)

        epsilon_identity = self.epsilon * torch.eye(C, device=I.device, dtype=I.dtype).view(1, C, C, 1, 1)
        S = cov_II + epsilon_identity  # (B, C, C, H, W)

        S = S.permute(0, 3, 4, 1, 2).reshape(-1, C, C)  # (B*H*W, C, C)

        cov_Ip_reshaped = cov_Ip.permute(0, 2, 3, 1).reshape(-1, C, 1)  # (B*H*W, C, 1)

        try:
            a = torch.linalg.solve(S, cov_Ip_reshaped)  # (B*H*W, C, 1)
        except RuntimeError:
            a = torch.linalg.lstsq(S, cov_Ip_reshaped).solution  # (B*H*W, C, 1)

        a = a.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        a_mean_I = torch.einsum("bchw,bchw->bhw", a, mean_I)  # (B, H, W)
        b = mean_p - a_mean_I.unsqueeze(1)  # (B, 1, H, W)

        mean_a = self.compute_mean(a)  # (B, C, H, W)
        mean_b = self.compute_mean(b)  # (B, 1, H, W)

        q = torch.einsum("bchw,bchw->bhw", mean_a, I) + mean_b.squeeze(1)  # (B, H, W)
        q = q.unsqueeze(1)  # (B, 1, H, W)

        return q
