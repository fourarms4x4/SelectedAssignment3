"""
Conditional Generative Adversarial Network (CGAN) on CIFAR-100
Conditioned on flower-related classes: Orchids and Roses

Analyzes generator and discriminator behavior over 10 training cycles.
Each cycle: prints discriminator outputs and saves 3 generated images.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────────
LATENT_DIM   = 100
EMBED_DIM    = 50
NUM_CLASSES  = 2          # orchids, roses  (re-mapped to 0 / 1)
IMAGE_SIZE   = 32
CHANNELS     = 3
BATCH_SIZE   = 64
LR           = 0.0002
BETAS        = (0.5, 0.999)
NUM_EPOCHS   = 50         # 10 reporting cycles  →  every 5 epochs
NUM_CYCLES   = 10
IMAGES_DIR   = "generated_images"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-100 fine-label indices for the two flower classes
# (can be verified via torchvision.datasets.CIFAR100.class_to_idx)
ORIGINAL_LABELS = {54: 0,   # orchids  → class 0
                   70: 1}   # roses    → class 1
CLASS_NAMES     = {0: "orchids", 1: "roses"}

os.makedirs(IMAGES_DIR, exist_ok=True)
print(f"Using device: {DEVICE}\n")


# ──────────────────────────────────────────────────────────────
# 1. Dataset – CIFAR-100, filtered to orchids & roses
# ──────────────────────────────────────────────────────────────
def get_flower_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),   # scale to [-1, 1]
    ])

    dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )

    # Filter indices that belong to orchids (54) or roses (70)
    targets = np.array(dataset.targets)
    indices = np.where(np.isin(targets, list(ORIGINAL_LABELS.keys())))[0]

    # Re-map original labels (54, 70) → (0, 1)
    dataset.targets = [
        ORIGINAL_LABELS[t] if t in ORIGINAL_LABELS else t
        for t in dataset.targets
    ]

    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=0, drop_last=True)

    print(f"Flower samples selected: {len(subset)}")
    label_counts = {0: 0, 1: 0}
    for idx in indices:
        label_counts[ORIGINAL_LABELS[targets[idx]]] += 1
    for k, v in label_counts.items():
        print(f"  {CLASS_NAMES[k]:10s} (re-mapped label {k}): {v} images")
    print()
    return loader


# ──────────────────────────────────────────────────────────────
# 2. Generator
# ──────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Input : noise vector (LATENT_DIM) + class label (integer)
    Output: RGB image  (CHANNELS × IMAGE_SIZE × IMAGE_SIZE)

    Architecture
    ────────────
    label  → Embedding(NUM_CLASSES, EMBED_DIM)
    [noise ‖ embed]  →  FC  →  reshape to (256, 4, 4)
    ConvTranspose × 3  →  (128, 8, 8)  →  (64, 16, 16)  →  (3, 32, 32)
    """
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)

        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBED_DIM, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),
        )

        self.conv = nn.Sequential(
            # 256 × 4 × 4  →  128 × 8 × 8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 × 8 × 8  →  64 × 16 × 16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 × 16 × 16  →  3 × 32 × 32
            nn.ConvTranspose2d(64, CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        embed = self.label_emb(labels)                # (B, EMBED_DIM)
        x     = torch.cat([noise, embed], dim=1)      # (B, LATENT_DIM + EMBED_DIM)
        x     = self.fc(x).view(-1, 256, 4, 4)        # (B, 256, 4, 4)
        return self.conv(x)                            # (B, 3, 32, 32)


# ──────────────────────────────────────────────────────────────
# 3. Discriminator
# ──────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Input : RGB image (3 × 32 × 32) + class label
    Output: scalar probability (real vs. fake)

    Architecture
    ────────────
    label  → Embedding → FC → reshape to (1, 32, 32)
    [image ‖ projected_label]  (4, 32, 32)
    Conv × 3  →  (64, 16, 16)  →  (128, 8, 8)  →  (256, 4, 4)
    Flatten  →  FC  →  Sigmoid
    """
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.label_proj = nn.Linear(EMBED_DIM, IMAGE_SIZE * IMAGE_SIZE)

        self.conv = nn.Sequential(
            # (3+1) × 32 × 32  →  64 × 16 × 16
            nn.Conv2d(CHANNELS + 1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 × 16 × 16  →  128 × 8 × 8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 × 8 × 8  →  256 × 4 × 4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, images, labels):
        embed = self.label_emb(labels)                              # (B, EMBED_DIM)
        proj  = self.label_proj(embed)                              # (B, 32*32)
        proj  = proj.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)            # (B, 1, 32, 32)
        x     = torch.cat([images, proj], dim=1)                    # (B, 4, 32, 32)
        x     = self.conv(x)
        return self.fc(x)                                           # (B, 1)


# ──────────────────────────────────────────────────────────────
# 4. Weight initialisation (DCGAN paper recommendation)
# ──────────────────────────────────────────────────────────────
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ──────────────────────────────────────────────────────────────
# 5. Visualise & save generated images
# ──────────────────────────────────────────────────────────────
def save_and_print_generated(generator, fixed_noise, fixed_labels,
                              cycle, epoch):
    """
    Generate 3 sample images (one orchid, one rose, one orchid again),
    save them to disk, and print pixel statistics as a proxy for
    visual 'output'. Returns the raw generated tensors.
    """
    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(fixed_noise, fixed_labels)  # (3, 3, 32, 32)
    generator.train()

    # De-normalise from [-1, 1] → [0, 1]
    imgs_np = (fake_imgs.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    imgs_np = imgs_np.transpose(0, 2, 3, 1)              # (3, 32, 32, 3)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(f"Cycle {cycle}  (Epoch {epoch}) – Generator Output",
                 fontsize=12)
    labels_str = [CLASS_NAMES[fixed_labels[i].item()] for i in range(3)]

    for i, ax in enumerate(axes):
        ax.imshow(imgs_np[i])
        ax.set_title(f"Label: {labels_str[i]}", fontsize=9)
        ax.axis("off")

    path = os.path.join(IMAGES_DIR, f"cycle_{cycle:02d}_epoch_{epoch:03d}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)

    # ── Print generator output statistics ──────────────────────
    print(f"  [Generator Output – 3 samples]")
    for i in range(3):
        img   = imgs_np[i]
        label = labels_str[i]
        print(f"    Image {i+1} (class={label}):  "
              f"mean={img.mean():.4f}  std={img.std():.4f}  "
              f"min={img.min():.4f}  max={img.max():.4f}  "
              f"→ saved to {path}")

    return fake_imgs


# ──────────────────────────────────────────────────────────────
# 6. Training loop
# ──────────────────────────────────────────────────────────────
def train():
    loader = get_flower_loader()

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion   = nn.BCELoss()
    opt_G       = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_D       = optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    # Fixed noise & labels used for consistent evaluation across cycles
    fixed_noise  = torch.randn(3, LATENT_DIM, device=DEVICE)
    fixed_labels = torch.tensor([0, 1, 0], device=DEVICE)   # orchid, rose, orchid

    epochs_per_cycle = NUM_EPOCHS // NUM_CYCLES   # = 5

    print("=" * 65)
    print("  CGAN Training   |   CIFAR-100 Flowers: Orchids & Roses")
    print(f"  Epochs: {NUM_EPOCHS}   Cycles: {NUM_CYCLES}   "
          f"Epochs/cycle: {epochs_per_cycle}")
    print("=" * 65, "\n")

    cycle = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_d_real, epoch_d_fake, epoch_loss_D, epoch_loss_G = [], [], [], []

        for real_imgs, labels in loader:
            real_imgs = real_imgs.to(DEVICE)
            labels    = labels.to(DEVICE)
            B         = real_imgs.size(0)

            real_lbl = torch.ones (B, 1, device=DEVICE)
            fake_lbl = torch.zeros(B, 1, device=DEVICE)

            # ── Train Discriminator ───────────────────────────
            opt_D.zero_grad()

            # Real images
            out_real = D(real_imgs, labels)
            loss_D_real = criterion(out_real, real_lbl)

            # Fake images
            noise      = torch.randn(B, LATENT_DIM, device=DEVICE)
            rand_labels = torch.randint(0, NUM_CLASSES, (B,), device=DEVICE)
            fake_imgs  = G(noise, rand_labels).detach()
            out_fake   = D(fake_imgs, rand_labels)
            loss_D_fake = criterion(out_fake, fake_lbl)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # ── Train Generator ───────────────────────────────
            opt_G.zero_grad()

            noise       = torch.randn(B, LATENT_DIM, device=DEVICE)
            rand_labels = torch.randint(0, NUM_CLASSES, (B,), device=DEVICE)
            fake_imgs   = G(noise, rand_labels)
            out_G       = D(fake_imgs, rand_labels)
            loss_G      = criterion(out_G, real_lbl)   # G wants D to say "real"
            loss_G.backward()
            opt_G.step()

            # Accumulate metrics
            epoch_d_real.append(out_real.mean().item())
            epoch_d_fake.append(out_fake.mean().item())
            epoch_loss_D.append(loss_D.item())
            epoch_loss_G.append(loss_G.item())

        # ── End-of-epoch averages ─────────────────────────────
        avg_d_real = np.mean(epoch_d_real)
        avg_d_fake = np.mean(epoch_d_fake)
        avg_loss_D = np.mean(epoch_loss_D)
        avg_loss_G = np.mean(epoch_loss_G)

        # ── Report at each cycle boundary ─────────────────────
        if epoch % epochs_per_cycle == 0:
            cycle += 1
            print(f"{'─'*65}")
            print(f"  CYCLE {cycle:2d}  |  Epoch {epoch}/{NUM_EPOCHS}")
            print(f"{'─'*65}")

            # ── Discriminator output ──────────────────────────
            print(f"  [Discriminator Output]")
            print(f"    D(real images) avg score : {avg_d_real:.4f}   "
                  f"(ideal → 1.0)")
            print(f"    D(fake images) avg score : {avg_d_fake:.4f}   "
                  f"(ideal → 0.0)")
            print(f"    Loss_D                   : {avg_loss_D:.4f}")
            print(f"    Loss_G                   : {avg_loss_G:.4f}")
            print()

            # ── Generator output (3 images) ───────────────────
            save_and_print_generated(G, fixed_noise, fixed_labels,
                                     cycle=cycle, epoch=epoch)
            print()

    print("=" * 65)
    print("  Training complete.")
    print(f"  Generated images saved in: ./{IMAGES_DIR}/")
    print("=" * 65)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
