import torch
import torch.nn.functional as F
from math import ceil


class Patcher:
    def __init__(self, h, w, patch_size, overlap):
        self.h, self.w = h, w
        self.patch_size = patch_size
        self.overlap = overlap

        self.stride = patch_size - overlap
        self.h_pad = self.stride * ceil((h - patch_size) / self.stride) + patch_size - h
        self.w_pad = self.stride * ceil((w - patch_size) / self.stride) + patch_size - w

        self.unfold = torch.nn.Unfold(
            kernel_size=(patch_size, patch_size), stride=self.stride
        )
        self.fold = torch.nn.Fold(
            output_size=(h + self.h_pad, w + self.w_pad),
            kernel_size=(patch_size, patch_size),
            stride=self.stride,
        )

    def extract_patches(self, x):
        assert x.ndim == 4
        x = F.pad(x, (0, self.w_pad, 0, self.h_pad), mode="reflect")
        B, C, _, _ = x.shape

        # (B, C, h_steps, w_steps, patch_size, patch_size)
        patches = x.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        patches = patches.permute(
            0, 2, 3, 1, 4, 5
        ).contiguous()  # (B, h_steps, w_steps, C, patch_size, patch_size)
        patches = patches.view(
            B, -1, C, self.patch_size, self.patch_size
        )  # (B, n_patches, C, patch_size, patch_size)
        return patches

    def merge_patches(self, patches):
        assert patches.ndim == 5
        B, N, C, _, _ = patches.shape

        # fold expects the patches tensor to have a shape (B, C * patch_size * patch_size, N)
        x = patches.permute(0, 2, 3, 4, 1).view(
            B, C * self.patch_size * self.patch_size, N
        )
        x = self.fold(x)  # (B, C, h + pad_h, w + pad_w)

        # as patches overlap we average the values of overlapping pixels
        weight_mask = 1 / self.fold(
            self.unfold(torch.ones(x.shape[-3:], device=patches.device))
        )
        x = x * weight_mask

        x = x[:, :, : self.h, : self.w]
        return x
