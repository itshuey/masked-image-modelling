import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchDiagnostics:
    """
    Utilities for extracting patch tokens from a ViT-style encoder and running
    simple patch-level diagnostics / probes.

    The encoder is assumed to have:
    - encoder.to_patch_embedding (nn.Sequential or list-like)
      where [0] creates patch vectors and [1:] maps to token embeddings
    - encoder.pos_embedding shaped (1, 1+P, D) (with optional CLS at index 0)
    - encoder.transformer mapping (B, P, D) -> (B, P, D)
    """

    def __init__(self, n_classes: int = 3, patch_size: int = 16):
        """
        Initialize patch diagnostic helpers.

        Args:
        - n_classes (int): Number of segmentation classes.
        - patch_size (int): Pixel patch size used by patchify-style ops.
        """
        self.n_classes = n_classes
        self.patch_size = patch_size

    @torch.no_grad()
    def encode_patch_tokens(
        self,
        encoder: nn.Module,
        imgs: torch.Tensor,
        device: torch.device,
        return_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Encode images into patch tokens using a frozen encoder.

        Args:
        - encoder (nn.Module): Vision transformer style encoder.
        - imgs (torch.Tensor): Image tensor of shape (B,3,H,W) or (3,H,W).
        - device (torch.device): Target device.
        - return_cpu (bool): If True, returns tokens on CPU.

        Returns:
        - tokens (torch.Tensor): Patch tokens of shape (B, P, D).
        """
        encoder.eval()

        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)  # (1,3,H,W)
        imgs = imgs.to(device)

        # Patchify + embed
        patches = encoder.to_patch_embedding[0](imgs)      # (B, P, patch_dim)
        tokens = encoder.to_patch_embedding[1:](patches)   # (B, P, D)

        # Add positional embeddings (skip CLS slot if present at index 0)
        P = tokens.shape[1]
        pos = encoder.pos_embedding[:, 1 : 1 + P].to(device)  # (1, P, D)
        tokens = encoder.transformer(tokens + pos)            # (B, P, D)

        if return_cpu:
            tokens = tokens.cpu()
        return tokens

    @torch.no_grad()
    def extract_patch_features(
        self,
        encoder: nn.Module,
        dataloader,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract patch tokens for a dataset.

        Args:
        - encoder (nn.Module): Frozen encoder.
        - dataloader: Yields (imgs, target) batches.
        - device (torch.device): Target device.
        - max_batches (int, optional): Stop after this many batches.

        Returns:
        - features (torch.Tensor): Shape (N, P, D) concatenated over batches.
        """
        encoder.eval()
        all_features = []

        for b, (imgs, _) in enumerate(dataloader):
            tokens = self.encode_patch_tokens(encoder, imgs, device, return_cpu=True)
            all_features.append(tokens)

            if max_batches is not None and (b + 1) >= max_batches:
                break

        return torch.cat(all_features, dim=0)  # (N, P, D)

    @torch.no_grad()
    def get_patch_tokens_single_image(
        self,
        encoder: nn.Module,
        img: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Extract patch tokens for a single image.

        Args:
        - encoder (nn.Module): Frozen encoder.
        - img (torch.Tensor): Image tensor of shape (3,H,W).
        - device (torch.device): Target device.

        Returns:
        - tokens (torch.Tensor): Patch tokens of shape (P, D) on CPU.
        """
        tokens = self.encode_patch_tokens(encoder, img, device, return_cpu=True)  # (1,P,D)
        return tokens[0]

    def patch_histogram(
        self,
        target: torch.Tensor,
        *,
        patch_size: Optional[int] = None,
        grid_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Convert a segmentation map into per-patch class counts.

        Exactly one of (patch_size, grid_size) should be provided:
        - patch_size: uses unfold with non-overlapping patch_size x patch_size
        - grid_size: partitions into (grid_size x grid_size) patches

        Args:
        - target (torch.Tensor): (B,H,W), (B,1,H,W), (H,W), or (1,H,W) integer labels.
        - patch_size (int, optional): Pixel patch size for unfolding.
        - grid_size (int, optional): Number of patches per side.

        Returns:
        - counts (torch.Tensor): Per-patch counts, shape (B, P, C).
        """
        if patch_size is not None and grid_size is not None:
            raise ValueError("Provide only one of patch_size or grid_size.")

        if patch_size is None and grid_size is None:
            raise ValueError("Provide patch_size or grid_size.")

        # Normalize shape to (B,H,W)
        if target.dim() == 2:
            target = target.unsqueeze(0)  # (1,H,W)
        if target.dim() == 3 and target.shape[0] == 1 and target.dtype != torch.float32:
            # ambiguous single-batch vs channel; handle below by ndim check
            pass
        if target.dim() == 4:
            # (B,1,H,W) -> (B,H,W)
            target = target[:, 0]

        target = target.long()
        B, H, W = target.shape
        C = self.n_classes

        if patch_size is not None:
            ps = patch_size
            # (B,1,H,W)
            tgt = target.unsqueeze(1)
            patches = tgt.unfold(2, ps, ps).unfold(3, ps, ps)  # (B,1,nH,nW,ps,ps)
            B_, _, nH, nW, ph, pw = patches.shape
            patches = patches.reshape(B_, nH * nW, ph * pw)     # (B,P,ps*ps)
        else:
            gs = grid_size
            ph, pw = H // gs, W // gs
            P = gs * gs
            # (B, gs, ph, gs, pw) -> (B, P, ph*pw)
            patches = (
                target.view(B, gs, ph, gs, pw)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view(B, P, ph * pw)
            )

        # Compute class counts (B, P, C)
        counts = torch.zeros(patches.shape[0], patches.shape[1], C, device=patches.device, dtype=torch.long)
        for k in range(C):
            counts[..., k] = (patches == k).sum(dim=-1)
        return counts

    def patchify_labels(self, seg: torch.Tensor, patch_size: Optional[int] = None) -> torch.Tensor:
        """
        Convert pixel-wise segmentation maps into patch-level labels by majority vote.

        Args:
        - seg (torch.Tensor): (B,H,W) or (B,1,H,W) integer segmentation labels.
        - patch_size (int, optional): Patch size. Defaults to self.patch_size.

        Returns:
        - labels (torch.Tensor): Patch labels of shape (B, P).
        """
        ps = self.patch_size if patch_size is None else patch_size
        counts = self.patch_histogram(seg, patch_size=ps)   # (B,P,C)
        return counts.argmax(dim=-1)

    def patch_class_fractions(self, target: torch.Tensor, grid_size: int = 8) -> torch.Tensor:
        """
        Compute per-patch class fractions by partitioning a segmentation map into a grid.

        Args:
        - target (torch.Tensor): (1,H,W) or (H,W) integer labels [0..C-1].
        - grid_size (int): Number of patches per side.

        Returns:
        - fracs (torch.Tensor): Fraction of each label per patch, shape (P, C).
        """
        counts = self.patch_histogram(target, grid_size=grid_size)  # (B,P,C)
        counts = counts[0].float()  # (P,C)
        return counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def variance_decomposition(self, features: torch.Tensor) -> Dict[str, float]:
        """
        Decompose patch feature variance into within-image and across-image components.

        Args:
        - features (torch.Tensor): Patch features of shape (N, P, D).

        Returns:
        - stats (dict): within_image_var, across_image_var, total_var as floats.
        """
        img_means = features.mean(dim=1)           # (N, D)
        var_across = img_means.var(dim=0).mean()
        var_within = features.var(dim=1).mean()
        return {
            "within_image_var": float(var_within.item()),
            "across_image_var": float(var_across.item()),
            "total_var": float((var_within + var_across).item()),
        }

    def cosine_similarity_vs_distance(self, features: torch.Tensor) -> Dict[int, float]:
        """
        Compute mean cosine similarity between patch tokens as a function of grid distance.

        Assumes patches form a square grid (P = g^2). 
        Distance is Manhattan distance between patch coordinates.

        Args:
        - features (torch.Tensor): Patch features of shape (N, P, D).

        Returns:
        - stats (dict): Maps integer distance -> mean cosine similarity (float).
        """
        N, P, D = features.shape
        g = int(round(math.sqrt(P)))
        if g * g != P:
            raise ValueError(f"Expected square number of patches, got P={P}.")

        # Normalize features for cosine similarity
        x = F.normalize(features, dim=-1)  # (N,P,D)

        # Precompute coordinates and distance matrix (P,P)
        coords = torch.stack(
            torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij"),
            dim=-1,
        ).view(P, 2)  # (P,2)
        dist = (coords[:, None, :] - coords[None, :, :]).abs().sum(dim=-1)  # (P,P)

        # Compute average cosine per distance (exclude self-pairs)
        stats: Dict[int, float] = {}
        # (N,P,P) cosine sim via batch matmul
        sim = torch.einsum("npd,nqd->npq", x, x)  # (N,P,P)

        for d in range(1, dist.max().item() + 1):
            mask = (dist == d)
            # mean over N and all (p,q) at that distance
            vals = sim[:, mask]  # (N, num_pairs_at_d)
            stats[int(d)] = float(vals.mean().item()) if vals.numel() > 0 else float("nan")

        return stats

    def run_patch_linear_probe(
        self,
        encoder: nn.Module,
        dataloader,
        device: torch.device,
        epochs: int = 5,
        lr: float = 1e-2,
        verbose=False
    ) -> float:
        """
        Train a frozen-encoder linear probe to predict patch-level segmentation labels.

        Returns:
        - acc (float): Final training accuracy over the last epoch.
        """
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        D = encoder.pos_embedding.shape[-1]
        probe = nn.Linear(D, self.n_classes).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        last_acc = 0.0

        for ep in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for img, seg in dataloader:
                img = img.to(device)
                seg = seg.to(device)

                labels = self.patchify_labels(seg)  # (B,P) on device? patchify uses ops on seg device
                tokens = self.encode_patch_tokens(encoder, img, device, return_cpu=False)  # (B,P,D)

                logits = probe(tokens)  # (B,P,C)

                loss = criterion(logits.reshape(-1, self.n_classes), labels.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

            acc = correct / max(1, total)
            last_acc = correct / total
            if verbose:
                print(f"[Probe] Epoch {ep+1:>2}/{epochs} | Loss {total_loss/len(dataloader):.4f} | Acc {last_acc:.4f}")

        print(f"==== PROBE FINAL ACC: {last_acc:.4f} ====")
        return float(acc)

    def run_full_diagnostics(
        self,
        name: str,
        encoder: nn.Module,
        dataloader,
        device: torch.device,
        max_images: int = 2000,
        verbose: bool = False
    ) -> Tuple[Dict[str, float], Dict[int, float]]:
        """
        Run variance and spatial cosine-similarity diagnostics on patch-level features.

        Returns:
        - var_stats (dict): Variance decomposition stats.
        - cos_stats (dict): Cosine similarity vs distance stats.
        """
        max_batches = math.ceil(max_images / dataloader.batch_size)
        features = self.extract_patch_features(
            encoder, dataloader, device, max_batches=max_batches
        )

        var_stats = self.variance_decomposition(features)
        cos_stats = self.cosine_similarity_vs_distance(features)

        if verbose:
            print(f"\n=== {name.upper()} ===")
            print(
                f"Variance: "
                f"within={var_stats['within_image_var']:.4f}, "
                f"across={var_stats['across_image_var']:.4f}, "
                f"total={var_stats['total_var']:.4f}"
            )
            print("Cosine similarity vs distance:")
            for d in sorted(cos_stats):
                print(f"  d={d}: {cos_stats[d]:.4f}")

        return var_stats, cos_stats


