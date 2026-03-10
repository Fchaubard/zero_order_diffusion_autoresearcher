"""
One-time data preparation for diffusion experiments.
Downloads ImageNet ILSVRC2012 and computes FID reference statistics.

Usage:
    python prepare.py                              # full prep (download + FID stats)
    python prepare.py --data-dir /path/to/imagenet  # use local ImageNet directory

Data and statistics are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import DataLoader, Dataset
from scipy import linalg

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

IMG_SIZE = 64            # image resolution (64x64)
IMG_CHANNELS = 3         # RGB
NUM_CLASSES = 1000       # ImageNet classes
TIME_BUDGET = 3600       # training time budget in seconds (1 hour)
FID_NUM_SAMPLES = 10000  # number of samples for FID evaluation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
STATS_DIR = os.path.join(CACHE_DIR, "stats")

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data(data_dir=None):
    """Download ImageNet ILSVRC2012 via HuggingFace datasets (cached, only downloads once)."""
    from datasets import load_dataset

    target_dir = data_dir if data_dir else DATA_DIR
    marker = os.path.join(target_dir, ".imagenet_ready")
    if os.path.exists(marker):
        print(f"Data: ImageNet already downloaded at {target_dir}")
        return

    os.makedirs(target_dir, exist_ok=True)

    print("Data: downloading ImageNet ILSVRC2012 from HuggingFace...")
    print("  (Requires accepting terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k)")
    print("  (Set HF_TOKEN env var or run `huggingface-cli login` first)")

    t0 = time.time()
    # Download both splits — cached by HF datasets library
    load_dataset("ILSVRC/imagenet-1k", split="train", cache_dir=target_dir)
    load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=target_dir)

    with open(marker, "w") as f:
        f.write("ready\n")

    t1 = time.time()
    print(f"Data: downloaded in {t1 - t0:.1f}s to {target_dir}")

# ---------------------------------------------------------------------------
# FID reference statistics
# ---------------------------------------------------------------------------

class InceptionFeatureExtractor:
    """Extract 2048-dim pool features from InceptionV3 for FID computation."""

    def __init__(self, device="cuda"):
        self.device = device
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = nn.Identity()  # Remove classification head, expose 2048-dim pool features
        model.eval()
        self.model = model.to(device)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_features(self, images):
        """Extract features from a batch of images in [0, 1] range."""
        images = self.transform(images.to(self.device))
        features = self.model(images)
        return features.cpu().numpy()


def compute_fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Frechet Inception Distance between two Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def compute_reference_stats(data_dir=None):
    """Compute and cache InceptionV3 statistics for validation images."""
    mu_path = os.path.join(STATS_DIR, "fid_mu.npy")
    sigma_path = os.path.join(STATS_DIR, "fid_sigma.npy")

    if os.path.exists(mu_path) and os.path.exists(sigma_path):
        print(f"FID stats: already computed at {STATS_DIR}")
        return

    os.makedirs(STATS_DIR, exist_ok=True)

    print("FID stats: computing reference statistics from validation set...")
    t0 = time.time()

    extractor = InceptionFeatureExtractor()
    val_loader = make_dataloader("val", batch_size=64, num_workers=4,
                                 infinite=False, data_dir=data_dir)

    all_features = []
    n_processed = 0
    for images, labels in val_loader:
        # images are in [-1, 1], convert to [0, 1]
        images_01 = (images + 1) / 2
        features = extractor.extract_features(images_01)
        all_features.append(features)
        n_processed += len(images)
        if n_processed >= FID_NUM_SAMPLES:
            break
        print(f"\r  Processed {n_processed}/{FID_NUM_SAMPLES} images", end="", flush=True)

    print()
    all_features = np.concatenate(all_features, axis=0)[:FID_NUM_SAMPLES]
    mu = np.mean(all_features, axis=0)
    sigma = np.cov(all_features, rowvar=False)

    np.save(mu_path, mu)
    np.save(sigma_path, sigma)

    t1 = time.time()
    print(f"FID stats: computed in {t1 - t0:.1f}s, saved to {STATS_DIR}")

    # Sanity check
    print(f"FID stats: mu shape={mu.shape}, sigma shape={sigma.shape}")
    print(f"FID stats: mu range=[{mu.min():.2f}, {mu.max():.2f}]")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class FlowMatching:
    """Flow Matching with optimal transport conditional paths."""

    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min

    def forward_sample(self, x0, t):
        """
        Interpolate from noise (t=0) to data (t=1).
        x0 is DATA, noise is sampled.
        Returns noisy sample x_t and velocity target.
        """
        data = x0
        noise = torch.randn_like(data)

        t_expanded = t.float().view(-1, 1, 1, 1).to(data.device)

        # OT conditional path: x_t = t * data + (1 - (1 - sigma_min) * t) * noise
        x_t = t_expanded * data + (1 - (1 - self.sigma_min) * t_expanded) * noise

        # Velocity target: v = data - (1 - sigma_min) * noise
        velocity = data - (1 - self.sigma_min) * noise

        return x_t, velocity

    def train_loss(self, model, x0, class_labels=None):
        """Compute flow matching MSE loss (teacher-forced, for backprop)."""
        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device=x0.device)
        x_t, velocity = self.forward_sample(x0, t)
        predicted_velocity = model(x_t, t, class_labels=class_labels)
        return F.mse_loss(predicted_velocity, velocity)

    @torch.no_grad()
    def sample(self, model, shape, device, num_steps=50, class_labels=None):
        """Euler ODE solver from noise (t=0) to data (t=1)."""
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = i / num_steps
            t_tensor = torch.full((shape[0],), t, device=device)
            velocity = model(x, t_tensor, class_labels=class_labels)
            x = x + velocity * dt

        return x

    def denoising_loss(self, model, x0, class_labels=None, denoising_steps=20, noise_seed=None):
        """
        Full ODE denoising loss for zero-order (SPSA) training.
        No teacher forcing: starts from pure Gaussian noise, runs the model
        denoising_steps times via Euler integration, and returns MSE between
        the denoised output and the clean data x0.

        Uses deterministic noise when noise_seed is provided, ensuring
        consistent loss evaluation for SPSA ±epsilon perturbation pairs.

        Returns: float (scalar loss value)
        """
        device = x0.device
        if noise_seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(noise_seed)
            noise = torch.randn(x0.shape, device=device, dtype=x0.dtype, generator=gen)
        else:
            noise = torch.randn_like(x0)

        x = noise
        dt = 1.0 / denoising_steps

        for i in range(denoising_steps):
            t = i / denoising_steps
            t_tensor = torch.full((x0.shape[0],), t, device=device)
            velocity = model(x, t_tensor, class_labels=class_labels)
            x = x + velocity * dt

        return F.mse_loss(x, x0).item()


class ImageNetDataset(Dataset):
    """Wraps HuggingFace ImageNet dataset with image transforms."""

    def __init__(self, split, transform, data_dir=None):
        from datasets import load_dataset
        hf_split = "train" if split == "train" else "validation"
        cache_dir = data_dir if data_dir else DATA_DIR
        self.ds = load_dataset("ILSVRC/imagenet-1k", split=hf_split, cache_dir=cache_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        image = self.transform(image)
        return image, label


def get_transform(split):
    """Get image transforms for train/val split. Output in [-1, 1]."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


def make_dataloader(split, batch_size, num_workers=4, infinite=True, data_dir=None):
    """
    ImageNet dataloader.
    If infinite=True, yields (images, labels, epoch) forever on GPU.
    If infinite=False, yields (images, labels) for one epoch on CPU.
    """
    assert split in ["train", "val"]
    transform = get_transform(split)
    dataset = ImageNetDataset(split, transform, data_dir=data_dir)

    if not infinite:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=True)

    # Infinite iterator with epoch tracking and GPU transfer
    def infinite_loader():
        epoch = 1
        while True:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                                num_workers=num_workers, pin_memory=True, drop_last=True,
                                persistent_workers=(num_workers > 0))
            for images, labels in loader:
                yield images.cuda(non_blocking=True), labels.cuda(non_blocking=True), epoch
            epoch += 1

    return infinite_loader()

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_fid(model, flow_matching, batch_size, num_classes=NUM_CLASSES,
                 num_samples=FID_NUM_SAMPLES, num_steps=50):
    """
    Frechet Inception Distance (FID): standard evaluation metric for generative models.
    Generates samples using the trained model and Euler ODE solver,
    extracts InceptionV3 pool features, and compares against precomputed
    reference statistics from real validation images.
    Lower FID = better quality and diversity.
    """
    mu_path = os.path.join(STATS_DIR, "fid_mu.npy")
    sigma_path = os.path.join(STATS_DIR, "fid_sigma.npy")
    ref_mu = np.load(mu_path)
    ref_sigma = np.load(sigma_path)

    device = next(model.parameters()).device
    extractor = InceptionFeatureExtractor(device=str(device))

    all_features = []
    n_generated = 0
    while n_generated < num_samples:
        current_batch = min(batch_size, num_samples - n_generated)
        # Generate class-conditional samples
        labels = torch.randint(0, num_classes, (current_batch,), device=device)
        shape = (current_batch, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
        samples = flow_matching.sample(model, shape, device, num_steps=num_steps,
                                       class_labels=labels)
        # Clamp to [-1, 1] and convert to [0, 1] for Inception
        samples = torch.clamp(samples, -1, 1)
        samples_01 = (samples + 1) / 2
        features = extractor.extract_features(samples_01)
        all_features.append(features)
        n_generated += current_batch
        print(f"\r  FID eval: generated {n_generated}/{num_samples} samples", end="", flush=True)

    print()
    all_features = np.concatenate(all_features, axis=0)[:num_samples]
    gen_mu = np.mean(all_features, axis=0)
    gen_sigma = np.cov(all_features, rowvar=False)
    fid = compute_fid_from_stats(ref_mu, ref_sigma, gen_mu, gen_sigma)
    return fid

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ImageNet data and FID stats for diffusion training")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to HF datasets cache (overrides default)")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data(data_dir=args.data_dir)
    print()

    # Step 2: Compute FID reference statistics
    compute_reference_stats(data_dir=args.data_dir)
    print()
    print("Done! Ready to train.")
