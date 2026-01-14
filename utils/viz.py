import os
import json
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable

@torch.no_grad()
def get_patch_tokens_single_image_vit(encoder, img, device, grid_size=8):
    """
    Extract patch tokens for a single image from *this repo's ViT encoder*.

    Assumes encoder has:
      - encoder.to_patch_embedding (indexable: [0] patchify, [1:] project)
      - encoder.pos_embedding with CLS at index 0
      - encoder.transformer that consumes (B,P,D)
    """
    encoder.eval()
    if img.dim() != 3 or img.shape[0] != 3:
        raise ValueError(f"img must be (3,H,W), got {tuple(img.shape)}")

    img = img.unsqueeze(0).to(device)  # (1,3,H,W)

    patches_ = encoder.to_patch_embedding[0](img)        # (1,P,patch_dim)
    tokens   = encoder.to_patch_embedding[1:](patches_)  # (1,P,D)

    # positional embeddings (skip CLS pos at 0)
    P = grid_size * grid_size
    pos_emb  = encoder.pos_embedding[:, 1:(P + 1)].to(device)
    if tokens.shape[1] != P:
        raise ValueError(f"Expected P={P} tokens, got {tokens.shape[1]}. Check grid_size.")
    if pos_emb.shape[1] != P:
        raise ValueError(f"Expected P={P} pos emb, got {pos_emb.shape[1]}. Check grid_size.")

    tokens   = tokens + pos_emb
    tokens   = encoder.transformer(tokens)               # (1,P,D)

    return tokens[0].detach().cpu()  # (P,D)

def patch_class_fractions(target, grid_size=8, n_classes=3):
    """
    Per-patch class fractions for an integer segmentation map.

    target: (H,W) or (1,H,W) tensor with labels [0..n_classes-1]
    returns: (P,n_classes)
    """
    if target.dim() == 3:
        if target.shape[0] != 1:
            raise ValueError("target must be (H,W) or (1,H,W) with integer labels.")
        target = target[0]
    if target.dim() != 2:
        raise ValueError(f"target must be (H,W) or (1,H,W). Got {tuple(target.shape)}")
    target = target.long()

    H, W = target.shape
    if H % grid_size != 0 or W % grid_size != 0:
        raise ValueError(f"H,W must be divisible by grid_size={grid_size}. Got H={H}, W={W}")

    ph, pw = H // grid_size, W // grid_size
    P = grid_size * grid_size

    tgt_p = target.view(grid_size, ph, grid_size, pw).permute(0, 2, 1, 3).contiguous()
    tgt_p = tgt_p.view(P, ph * pw)  # (P, Npix)

    # Vectorized: (P,Npix)->(P,Npix,C)->mean over pixels => (P,C)
    fracs = F.one_hot(tgt_p, num_classes=n_classes).float().mean(dim=1)
    return fracs

def choose_reference_patches(target, grid_size=8, n_classes=3, fg_label=0, bg_label=1, boundary_label=2,
                            interior_boundary_threshold=0.2):
    """
    Picks:
      - bg corner patch (0,0)
      - fg "interior" patch: max fg fraction among low-boundary patches (fallback: global max fg)
      - boundary patch: max boundary fraction
    """
    fracs = patch_class_fractions(target, grid_size=grid_size, n_classes=n_classes)

    corner_idx = 0
    bd_idx = int(fracs[:, boundary_label].argmax().item())

    boundary_fr = fracs[:, boundary_label]
    fg_fr = fracs[:, fg_label]
    mask = boundary_fr <= interior_boundary_threshold
    if mask.any():
        fg_idx = int(fg_fr.masked_fill(~mask, float("-inf")).argmax().item())
    else:
        fg_idx = int(fg_fr.argmax().item())

    return {"bg_corner": corner_idx, "fg_interior": fg_idx, "boundary": bd_idx}

def cosine_sim_grid(tokens, ref_idx, grid_size=8, eps=1e-8):
    """
    tokens: (P,D)
    ref_idx: int
    returns: (grid_size, grid_size) tensor on CPU
    """
    P = grid_size * grid_size
    if tokens.dim() != 2 or tokens.shape[0] != P:
        raise ValueError(f"tokens must be (P,D) with P={P}. Got {tuple(tokens.shape)}")
    if not (0 <= ref_idx < P):
        raise ValueError(f"ref_idx must be in [0,{P-1}], got {ref_idx}")

    t = F.normalize(tokens, dim=-1, eps=eps)
    ref = t[ref_idx:ref_idx+1]
    sims = (t * ref).sum(dim=-1)
    return sims.view(grid_size, grid_size).detach().cpu()

def save_figure_with_metadata(fig, savepath, metadata=None, dump_metadata=True):
    if savepath is None:
        return

    print(f"Saving figure to {savepath}")
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    fig.savefig(savepath, bbox_inches="tight")

    if dump_metadata and metadata is not None:
        meta_path = os.path.splitext(savepath)[0] + ".json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

def plot_patch_similarity_figure(
    encoders: dict,
    dataset,
    idx: int,
    device,
    *,
    grid_size: int = 8,
    n_classes: int = 3,
    fg_label: int = 0,
    bg_label: int = 1,
    boundary_label: int = 2,
    include_gt: bool = True,
    fixed_vmin: float = -0.5,
    fixed_vmax: float = 1.0,
    savepath: str | None = None,
    dump_metadata: bool = True,
):
    """
    Visualize patch-token cosine similarity heatmaps for multiple encoders.

    Assumes dataset[idx] returns (img, target), where:
        img: (3,H,W) torch.Tensor
        target: (H,W) or (1,H,W) integer labels in [0..n_classes-1]

    Args:
      encoders: dict name -> encoder
      dataset: indexable dataset
      idx: image index
      device: torch device or string
      grid_size: patch grid size per side (P=grid_size^2)
      include_gt: include target mask in header
      fixed_vmin/vmax: shared color scale for similarity heatmaps
      savepath: if provided, saves figure to disk
      dump_metadata: if True and savepath is provided, dumps a JSON next to the figure

    Returns:
      refs: dict of selected reference patches (keys like bg_corner/fg_interior/boundary)
    """
    # --- load sample ---
    img, target = dataset[idx]
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Expected img to be torch.Tensor, got {type(img)}")
    if img.dim() != 3 or img.shape[0] != 3:
        raise ValueError(f"Expected img shape (3,H,W), got {tuple(img.shape)}")

    # --- choose reference patches ---
    refs = choose_reference_patches(
        target,
        grid_size=grid_size,
        n_classes=n_classes,
        fg_label=fg_label,
        bg_label=bg_label,
        boundary_label=boundary_label,
    )
    ref_keys = [k for k in ("bg_corner", "fg_interior", "boundary") if k in refs]
    if len(ref_keys) == 0:
        raise RuntimeError("No reference patches were selected; check target labels and config.")

    # --- layout ---
    enc_names = list(encoders.keys())
    n_rows = len(ref_keys)
    n_cols = len(enc_names)

    header_cols = 2 if include_gt else 1
    fig = plt.figure(figsize=(3.8 * max(n_cols, header_cols), 2.6 * (n_rows + 1)), dpi=160)
    gs = fig.add_gridspec(n_rows + 1, max(n_cols, header_cols), hspace=0.25, wspace=0.15)

    # --- header: input image (+ ref boxes) ---
    ax0 = fig.add_subplot(gs[0, 0])
    img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
    ax0.imshow(img_np)
    ax0.set_title("Input (refs in red)")
    ax0.axis("off")

    H, W = img_np.shape[:2]
    if H % grid_size != 0 or W % grid_size != 0:
        raise ValueError(f"Image H,W must be divisible by grid_size={grid_size}. Got H={H}, W={W}")
    ph, pw = H // grid_size, W // grid_size

    for rk in ref_keys:
        pidx = refs[rk]
        r, c = pidx // grid_size, pidx % grid_size
        ax0.add_patch(plt.Rectangle((c * pw, r * ph), pw, ph, fill=False, edgecolor="red", linewidth=2))
        label = rk.replace("_", " ")
        txt = ax0.text(c * pw + 2, r * ph + 12, label, color="red", fontsize=9, weight="bold")
        txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="white")])

    # --- header: target mask ---
    if include_gt:
        ax1 = fig.add_subplot(gs[0, 1])
        tgt = target
        if isinstance(tgt, torch.Tensor) and tgt.dim() == 3 and tgt.shape[0] == 1:
            tgt = tgt[0]
        if not isinstance(tgt, torch.Tensor) or tgt.dim() != 2:
            raise ValueError("target must be a tensor of shape (H,W) or (1,H,W) with integer labels.")
        ax1.imshow(tgt.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=n_classes - 1)
        ax1.set_title("Target mask")
        ax1.axis("off")

    # --- heatmaps ---
    # Precompute tokens per encoder once (cheaper than recomputing per row).
    tokens_by_enc = {}
    for name, enc in encoders.items():
        tokens_by_enc[name] = get_patch_tokens_single_image_vit(enc, img, device, grid_size=grid_size)

    last_im = None
    for ri, rk in enumerate(ref_keys, start=1):
        ref_idx = refs[rk]
        for ci, enc_name in enumerate(enc_names):
            ax = fig.add_subplot(gs[ri, ci])
            sim = cosine_sim_grid(tokens_by_enc[enc_name], ref_idx, grid_size=grid_size)

            last_im = ax.imshow(sim, vmin=fixed_vmin, vmax=fixed_vmax, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])

            if ri == 1:
                ax.set_title(enc_name, fontsize=11, weight="bold")
            if ci == 0:
                ax.set_ylabel(rk.replace("_", " "), fontsize=10, weight="bold")

            # Mark reference cell
            rr, cc = ref_idx // grid_size, ref_idx % grid_size
            ax.add_patch(plt.Rectangle((cc - 0.5, rr - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))

            # Put a colorbar only on the last column for each row
            if ci == n_cols - 1 and last_im is not None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="6%", pad=0.06)
                cb = plt.colorbar(last_im, cax=cax)
                cb.set_label("cos sim", fontsize=9)

    fig.suptitle("Patch token cosine similarity (reference patch vs all patches)", y=0.99, fontsize=13, weight="bold")
    meta = {
        "idx": idx,
        "grid_size": grid_size,
        "n_classes": n_classes,
        "labels": {"fg": fg_label, "bg": bg_label, "boundary": boundary_label},
        "refs": refs,
        "encoders": enc_names,
        "fixed_vmin": fixed_vmin,
        "fixed_vmax": fixed_vmax,
    }

    save_figure_with_metadata(fig, savepath, metadata=meta, dump_metadata=dump_metadata)

    display(fig)
    plt.close(fig)
    return refs
