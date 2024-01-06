import torch
import torch.nn.functional as F


class Patcher:
    def __init__(self, h, w, patch_size, overlap):
        self.h = h
        self.w = w
        self.patch_size = patch_size
        self.overlap = overlap

        self.stride = patch_size - overlap
        self.h_pad = self.stride * ceil((h - patch_size) / self.stride) + patch_size - h
        self.w_pad = self.stride * ceil((w - patch_size) / self.stride) + patch_size - w

        self.unfold = torch.nn.Unfold(
            kernel_size=(patch_size, patch_size),
            stride=self.stride
        )
        self.fold = torch.nn.Fold(
            output_size=(h + self.h_pad, w + self.w_pad),
            kernel_size=(patch_size, patch_size),
            stride=self.stride
        )

    def extract_patches(self, x):
        bs = len(x)
        x = F.pad(x, (0, self.w_pad, 0, self.h_pad), mode="reflect")
        patches = self.unfold(x).view(bs, self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2)
        return patches
    
    def merge_patches(self, patches):
        bs = len(patches)
        average_mask = (1 / self.fold(self.unfold(torch.ones(1, self.h + self.h_pad, self.w + self.w_pad, device=patches.device))))

        x_restored = self.fold(patches.permute(0, 2, 3, 1).view(bs, self.patch_size * self.patch_size, -1))
        x_restored = x_restored * average_mask
        x_restored = x_restored[:, :, :self.h, :self.w]

        return x_restored
