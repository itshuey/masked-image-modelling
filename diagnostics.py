import argparse
import os
import torch

from models.vit import ViT
from utils.diagnostics import PatchDiagnostics
from utils.configs import configs
from utils.viz import plot_patch_similarity_figure

def build_encoders(device, config, ckpt_dir="intermediate_checkpoints"):
    ckpts = {
        "pretrained_only": os.path.join(ckpt_dir, "encoder_pretrained_only.pth"),
        "cls_intermediate_intel": os.path.join(ckpt_dir, "encoder_cls_intel.pth"),
        "cls_intermediate_pets": os.path.join(ckpt_dir, "encoder_cls_pets.pth"),
        "spatial_intermediate": os.path.join(ckpt_dir, "encoder_spatial_intermediate.pth"),
    }

    encoders = {}
    for name, path in ckpts.items():
        if not os.path.exists(path):
            continue
        enc = ViT(
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            dim=config["dim"],
            depth=config["depth"],
            heads=config["heads"],
            mlp_dim=config["mlp_dim"]
        ).to(device)
        enc.load_state_dict(torch.load(path, map_location=device), strict=False)
        enc.eval()
        encoders[name] = enc

    if not encoders:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}.")
    return encoders

def build_dataloader(batch_size=64, image_size=128, num_workers=2):
    from torch.utils.data import DataLoader
    from utils.utils import AugmentedOxfordIIITPet, finetune_transforms

    pets_transform = finetune_transforms(image_size)
    ds = AugmentedOxfordIIITPet(
        root="data",
        split="trainval",
        target_types="segmentation",
        download=True,
        **pets_transform,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='vit_4M_finetune', help='Configuration of intermediate model.')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ckpt-dir", default="intermediate_checkpoints")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--do-probe", action="store_true")
    parser.add_argument("--probe-epochs", type=int, default=5)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--viz-patch-sim", action="store_true")
    parser.add_argument("--viz-idx", type=int, default=0)
    parser.add_argument("--viz-savepath", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config = configs[args.config]
    encoder_map = build_encoders(device=device, config=config, ckpt_dir=args.ckpt_dir)
    dataloader = build_dataloader(batch_size=args.batch_size, image_size=args.image_size)

    diag = PatchDiagnostics(n_classes=3, patch_size=16)

    results = []
    for name, enc in encoder_map.items():
        var_stats, cos_stats = diag.run_full_diagnostics(name, enc, dataloader, device, max_images=args.max_images, verbose=args.verbose)
        probe_acc = None
        if args.do_probe:
            probe_acc = diag.run_patch_linear_probe(enc, dataloader, device, epochs=args.probe_epochs, lr=args.probe_lr, verbose=args.verbose)
        results.append((name, var_stats, cos_stats, probe_acc))

    print("\n================ SUMMARY ================")
    header = (
        f"{'model':<24} "
        f"{'total_var':>10} {'within':>10} {'across':>10} "
        f"{'cos@1':>8} {'min_cos':>10}"
    )
    if args.do_probe:
        header += f" {'probe':>8}"
    print(header)

    for name, var_stats, cos_stats, probe_acc in results:
        min_d = min(cos_stats, key=lambda d: cos_stats[d])

        row = (
            f"{name:<24} "
            f"{var_stats['total_var']:>10.4f} "
            f"{var_stats['within_image_var']:>10.4f} "
            f"{var_stats['across_image_var']:>10.4f} "
            f"{cos_stats.get(1, float('nan')):>8.4f} "
            f"{cos_stats[min_d]:>10.4f}"
        )
        if args.do_probe:
            row += f" {probe_acc:>8.4f}"

        print(row)

    # highlight best probe
    best = max((r for r in results if r[3] is not None), key=lambda r: r[3], default=None)
    if best is not None:
        print(f"\nBEST PROBE: {best[0]}  acc={best[3]:.4f}")

    if args.viz_patch_sim:
        plot_patch_similarity_figure(
            encoders=encoder_map,
            dataset=dataloader.dataset,
            idx=args.viz_idx,
            device=device,
            grid_size=config["image_size"] // config["patch_size"],
            n_classes=3,
            include_gt=True,
            savepath=getattr(args, "viz_savepath", None),
        )

if __name__ == "__main__":
    main()
