import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from models.vit import ViT
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.diagnostics import PatchDiagnostics
from utils.utils import AugmentedOxfordIIITPet, finetune_transforms
from utils.configs import configs

# Data loaders
def build_pets_seg_loader(batch_size=64, image_size=128, num_workers=2):
    """
    Build Oxford-IIIT Pets segmentation loader for spatial intermediate training.
    """
    pets_transform = finetune_transforms(image_size)
    pets_trainset = AugmentedOxfordIIITPet(
        root="data",
        split="trainval",
        target_types="segmentation",
        download=True,
        **pets_transform,
    )
    return DataLoader(pets_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def build_intel_cls_loader(batch_size=64, image_size=128, num_workers=2):
    """
    Build Intel classification loader for CLS intermediate training.
    """
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    intel_transform = T.Compose([
        T.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(contrast=0.3),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    INTEL_ROOT = "data/seg_train/seg_train"
    intel_trainset = datasets.ImageFolder(root=INTEL_ROOT, transform=intel_transform)
    return DataLoader(intel_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def build_pets_cls_loader(batch_size=64, image_size=128, num_workers=2):
    """
    Build Oxford-IIIT Pets classification loader for CLS intermediate (pets) training.
    """
    tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="category",
        download=True,
        transform=tfm
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Training obj
def run_cls_intermediate_intel(encoder, intel_train_loader, device, epochs=20, lr=8e-4, num_classes=6):
    """
    Run CLS classification intermediate fine-tuning on Intel dataset.
    """
    encoder = encoder.to(device)
    encoder.train()

    encoder.head = nn.Linear(encoder.pos_embedding.shape[-1], num_classes).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        running_loss = 0.0
        for imgs, labels in intel_train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = encoder(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[CLS-Intel] Epoch {ep+1}/{epochs} | Loss {running_loss/len(intel_train_loader):.4f}")

    return encoder


def run_cls_intermediate_pets(encoder, pets_cls_loader, device, epochs=20, lr=1e-3, num_classes=37):
    """
    Run CLS classification intermediate fine-tuning on Oxford Pets categories.
    """
    encoder = encoder.to(device)
    encoder.train()

    encoder.head = nn.Linear(encoder.pos_embedding.shape[-1], num_classes).to(device)
    optimizer = optim.AdamW(encoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        running_loss = 0.0
        for imgs, labels in pets_cls_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = encoder(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[CLS-Pets] Epoch {ep+1}/{epochs} | Loss {running_loss/len(pets_cls_loader):.4f}")

    return encoder


def run_spatial_intermediate_pets(encoder, pets_seg_loader, device, diag: PatchDiagnostics, epochs=20, lr=5e-4):
    """
    Run patchwise spatial intermediate fine-tuning on Pets segmentation.
    """
    encoder = encoder.to(device)
    encoder.train()

    # Linear head over patch tokens
    D = encoder.pos_embedding.shape[-1]
    head = nn.Linear(D, diag.n_classes).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        running_loss = 0.0
        for imgs, seg in pets_seg_loader:
            imgs, seg = imgs.to(device), seg.to(device)
            labels = diag.patchify_labels(seg)  # (B,P)

            # We need tokens on device, and we need gradients => no @torch.no_grad
            patches = encoder.to_patch_embedding[0](imgs)
            tokens  = encoder.to_patch_embedding[1:](patches)
            P = tokens.shape[1]
            pos = encoder.pos_embedding[:, 1:1+P].to(device)
            tokens = encoder.transformer(tokens + pos)  # (B,P,D)

            logits = head(tokens)  # (B,P,C)
            loss = criterion(logits.view(-1, diag.n_classes), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[Spatial-Pets] Epoch {ep+1}/{epochs} | Loss {running_loss/len(pets_seg_loader):.4f}")

    return encoder

def main():
    """
    Train multiple intermediate objectives starting from a pretrained encoder,
    and save checkpoints for later diagnostics.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='vit_4M_finetune', help='Configuration of pre-trained model.')
    parser.add_argument("--pretrained-path", type=str, default="weights/encoder_vit_4M_pretrain_200K.pth")
    parser.add_argument("--save-dir", type=str, default="intermediate_checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    config = configs[args.config]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # loaders
    pets_seg_loader = build_pets_seg_loader(args.batch_size, args.image_size)
    intel_cls_loader = build_intel_cls_loader(args.batch_size, args.image_size)
    pets_cls_loader = build_pets_cls_loader(args.batch_size, args.image_size)

    diag = PatchDiagnostics(n_classes=3, patch_size=16)

    # helper to init from pretrained each time
    def fresh_from_pretrained():
        enc = ViT(
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            dim=config["dim"],
            depth=config["depth"],
            heads=config["heads"],
            mlp_dim=config["mlp_dim"]
        ).to(device)
        enc.load_state_dict(torch.load(args.pretrained_path, map_location=device), strict=True)
        return enc

    # 1) pretrained only
    enc = fresh_from_pretrained()
    torch.save(enc.state_dict(), os.path.join(args.save_dir, "encoder_pretrained_only.pth"))
    print("[✓] Saved pretrained_only\n")

    # 2) cls intermediate (intel)
    print("Starting classification intermediate fine-tuning on the Intel dataset")
    enc = fresh_from_pretrained()
    enc = run_cls_intermediate_intel(enc, intel_cls_loader, device, epochs=args.epochs)
    torch.save(enc.state_dict(), os.path.join(args.save_dir, "encoder_cls_intel.pth"))
    print("[✓] Saved cls_intermediate_intel\n")

    # 3) cls intermediate (pets)
    print("Starting classification intermediate fine-tuning on the Oxford Pets dataset")
    enc = fresh_from_pretrained()
    enc = run_cls_intermediate_pets(enc, pets_cls_loader, device, epochs=args.epochs)
    torch.save(enc.state_dict(), os.path.join(args.save_dir, "encoder_cls_pets.pth"))
    print("[✓] Saved cls_intermediate_pets\n")

    # 4) spatial intermediate (pets segmentation)
    print("Starting spatial intermediate fine-tuning on the Pets dataset")
    enc = fresh_from_pretrained()
    enc = run_spatial_intermediate_pets(enc, pets_seg_loader, device, diag, epochs=args.epochs)
    torch.save(enc.state_dict(), os.path.join(args.save_dir, "encoder_spatial_intermediate.pth"))
    print("[✓] Saved spatial_intermediate")

if __name__ == "__main__":
    main()
