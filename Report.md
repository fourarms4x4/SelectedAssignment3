# Assignment 3 – Conditional Generative Adversarial Network (CGAN)
**Course:** AIE418  
**Date:** March 9, 2026

---

## 1. Objective

Implement a **Conditional Generative Adversarial Network (CGAN)** in Python using the **CIFAR-100** dataset. The model is conditioned on two flower-related classes — **orchids** and **roses** — and trained to generate realistic images belonging to either class. Generator and discriminator behavior are analyzed across **10 training cycles**.

---

## 2. Dataset

**Dataset:** CIFAR-100 (Canadian Institute For Advanced Research, 100 classes)  
**Source:** `torchvision.datasets.CIFAR100` (auto-downloaded)  
**Image size:** 32 × 32 × 3 (RGB)  
**Training split used:** 50,000 images total → filtered to 2 classes

### Selected Classes

| Class Name | CIFAR-100 Fine Label | Re-mapped Label |
|------------|---------------------|-----------------|
| Orchids    | 54                  | 0               |
| Roses      | 70                  | 1               |

Each class has **500 training images** in CIFAR-100, giving a combined subset of **1,000 images**.

### Preprocessing

Images are normalized to the range $[-1, 1]$ using:

$$x' = \frac{x - 0.5}{0.5}$$

applied channel-wise (mean = 0.5, std = 0.5 per channel). This matches the Tanh output range of the Generator.

---

## 3. Model Architecture

### 3.1 Generator

The Generator takes a **random noise vector** and a **class label**, and produces a synthetic 32×32 RGB image conditioned on the label.

| Stage | Operation | Input Shape | Output Shape |
|-------|-----------|-------------|--------------|
| Label Embedding | `nn.Embedding(2, 50)` | (B,) | (B, 50) |
| Concatenate | `[noise ‖ embed]` | (B,100)+(B,50) | (B, 150) |
| Fully Connected | Linear + BatchNorm1d + ReLU | (B, 150) | (B, 4096) |
| Reshape | — | (B, 4096) | (B, 256, 4, 4) |
| ConvTranspose 1 | 256→128, k=4, s=2, p=1 + BN + ReLU | (B, 256, 4, 4) | (B, 128, 8, 8) |
| ConvTranspose 2 | 128→64, k=4, s=2, p=1 + BN + ReLU | (B, 128, 8, 8) | (B, 64, 16, 16) |
| ConvTranspose 3 | 64→3, k=4, s=2, p=1 + **Tanh** | (B, 64, 16, 16) | (B, 3, 32, 32) |

**Key design choices:**
- Class information is injected via a learned embedding, concatenated with noise before the fully connected layer.
- Tanh activation ensures pixel values lie in $[-1, 1]$, consistent with normalized inputs.
- BatchNorm is applied after each transposed convolution to stabilize training.

### 3.2 Discriminator

The Discriminator receives a **real or generated image** and a **class label**, and outputs a scalar probability indicating whether the image is real.

| Stage | Operation | Input Shape | Output Shape |
|-------|-----------|-------------|--------------|
| Label Embedding | `nn.Embedding(2, 50)` | (B,) | (B, 50) |
| Label Projection | Linear(50 → 1024) + reshape | (B, 50) | (B, 1, 32, 32) |
| Concatenate | `[image ‖ label_map]` | (B,3,32,32)+(B,1,32,32) | (B, 4, 32, 32) |
| Conv 1 | 4→64, k=4, s=2, p=1 + LeakyReLU(0.2) | (B, 4, 32, 32) | (B, 64, 16, 16) |
| Conv 2 | 64→128, k=4, s=2, p=1 + BN + LeakyReLU(0.2) | (B, 64, 16, 16) | (B, 128, 8, 8) |
| Conv 3 | 128→256, k=4, s=2, p=1 + BN + LeakyReLU(0.2) | (B, 128, 8, 8) | (B, 256, 4, 4) |
| Flatten + FC | Linear(4096 → 1) + **Sigmoid** | (B, 4096) | (B, 1) |

**Key design choices:**
- The label is projected into a spatial map (1×32×32) and concatenated to the image as an additional channel, so the discriminator can evaluate class consistency.
- LeakyReLU (slope 0.2) is used instead of ReLU to avoid the dying-gradient problem.
- No BatchNorm in the first convolutional layer (standard DCGAN practice).

### 3.3 Weight Initialization

All convolutional and batch normalization weights are initialized following the DCGAN paper:
- Conv weights: $\mathcal{N}(0,\ 0.02)$
- BatchNorm weights: $\mathcal{N}(1,\ 0.02)$,  bias = 0

---

## 4. Training Procedure

### 4.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent dimension | 100 |
| Embedding dimension | 50 |
| Batch size | 64 |
| Learning rate | 0.0002 |
| Adam β₁ | 0.5 |
| Adam β₂ | 0.999 |
| Total epochs | 50 |
| Reporting cycles | 10 (every 5 epochs) |
| Loss function | Binary Cross-Entropy (BCE) |

### 4.2 Adversarial Loss

Both networks are trained with **Binary Cross-Entropy Loss**:

$$\mathcal{L}_D = -\frac{1}{2}\left[\mathbb{E}[\log D(x, y)] + \mathbb{E}[\log(1 - D(G(z, y), y))]\right]$$

$$\mathcal{L}_G = -\mathbb{E}[\log D(G(z, y), y)]$$

Where:
- $x$ = real image, $z$ = noise vector, $y$ = class label
- $D(x, y)$ = discriminator output for a real image with label $y$
- $G(z, y)$ = generator output conditioned on noise and label $y$

The Generator is trained to **maximize** $D(G(z,y),y)$ (fool the discriminator), while the Discriminator is trained to correctly classify real images as real and generated images as fake.

### 4.3 Training Steps per Iteration

1. **Discriminator update:**
   - Compute $\mathcal{L}_{D,\text{real}}$ on a batch of real images.
   - Generate a batch of fake images with random labels.
   - Compute $\mathcal{L}_{D,\text{fake}}$ on the detached fake images.
   - Total: $\mathcal{L}_D = 0.5 \times (\mathcal{L}_{D,\text{real}} + \mathcal{L}_{D,\text{fake}})$
   - Backpropagate and update $D$.

2. **Generator update:**
   - Generate a new batch of fake images with random labels.
   - Pass them through $D$ with target label = "real" (1).
   - Backpropagate $\mathcal{L}_G$ and update $G$.

---

## 5. Analysis of Generator and Discriminator Behavior

### 5.1 What Is Tracked per Cycle

At the end of each 5-epoch cycle, the following metrics are printed:

| Metric | Interpretation |
|--------|---------------|
| `D(real)` average score | How confidently D identifies real images as real. Ideal → 1.0 |
| `D(fake)` average score | How confidently D rejects generated images. Ideal → 0.0 |
| `Loss_D` | Discriminator's total BCE loss |
| `Loss_G` | Generator's BCE loss (how well it fools D) |

### 5.2 Generator Output per Cycle

At each of the 10 cycles, **3 fixed-seed images** are generated (orchid, rose, orchid) using the **same noise vector and labels** throughout training. This allows direct visual comparison of improvement across cycles.

For each image, the following statistics are printed:

- **mean** – average pixel brightness (higher = brighter image)
- **std** – pixel value spread (higher = more contrast/texture)
- **min / max** – pixel value range

Generated images are saved to `generated_images/cycle_XX_epoch_YYY.png`.

### 5.3 Expected Training Dynamics

| Phase | D(real) | D(fake) | Loss_D | Loss_G | Interpretation |
|-------|---------|---------|--------|--------|---------------|
| Early | ~0.7–0.9 | ~0.4–0.6 | moderate | high | D learns fast; G produces noise |
| Mid | ~0.6–0.8 | ~0.2–0.4 | decreasing | decreasing | G begins to structure outputs |
| Late | ~0.5–0.6 | ~0.3–0.5 | stabilizing ~0.5 | stabilizing ~0.7 | Nash equilibrium approximated |

A well-trained CGAN reaches a state where `D(real) ≈ D(fake) ≈ 0.5`, meaning the discriminator can no longer reliably distinguish real from generated images. In practice, the Generator's samples should progressively develop structure resembling flower images across cycles.

---

## 6. Code Structure

```
cgan_cifar100_flowers.py
│
├── Hyperparameters (global constants)
├── get_flower_loader()        # Loads & filters CIFAR-100 to orchids/roses
├── Generator (nn.Module)      # Noise + label → 32×32 RGB image
├── Discriminator (nn.Module)  # Image + label → real/fake probability
├── weights_init()             # DCGAN weight initialization
├── save_and_print_generated() # Generates & saves 3 images; prints stats
└── train()                    # Main training loop with 10-cycle reporting
```

---

## 7. Dependencies

| Library | Purpose |
|---------|---------|
| `torch` | Neural network framework, autograd |
| `torchvision` | CIFAR-100 dataset, transforms |
| `numpy` | Array operations, metric aggregation |
| `matplotlib` | Image grid rendering and saving |

Install with:
```bash
pip install torch torchvision matplotlib numpy
```

---

## 8. How to Run

```bash
python cgan_cifar100_flowers.py
```

**Output:**
- Console: Per-cycle discriminator scores and generator pixel statistics for 3 images
- Disk: `generated_images/cycle_01_epoch_005.png` through `cycle_10_epoch_050.png`

---

## 9. Summary

A CGAN was implemented from scratch in PyTorch and trained on the flower subset of CIFAR-100 (orchids and roses). The architecture follows DCGAN conventions with class conditioning via learned embeddings injected at both the Generator input and the Discriminator input (as a projected spatial channel). Training is monitored over 10 equidistant cycles by printing discriminator output scores and generator image statistics, and saving representative image grids for qualitative evaluation.
