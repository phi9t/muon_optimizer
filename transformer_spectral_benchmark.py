"""
MNIST Vision Transformer Spectral Benchmark: Muon+Aux vs AdamW
Tracks train loss, attention entropy, and singular values of Q attention projections.
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from rich.console import Console
from rich.table import Table

from muon_optimizer import SingleDeviceMuonWithAuxAdam, create_muon_param_groups

# Configure logger
logger = logging.getLogger("transformer_spectral")

class Attention(nn.Module):
    """Multi-head self-attention module."""
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        # Project and split into q, k, v
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calculate attention maps
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Weighted sum and output projection
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out), attn

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, attn_map = self.attn(self.norm1(x))
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, attn_map

class SimpleViT(nn.Module):
    """Simple Vision Transformer architecture."""
    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        channels: int = 1
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        
        # Patch projection
        self.patch_to_embed = nn.Linear(channels * patch_size * patch_size, embed_dim)
        
        # Learnable parameters (using _embed suffix for AdamW routing via create_muon_param_groups)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)])
        
        # Output layers
        self.norm = nn.LayerNorm(embed_dim)
        # using 'head' name for AdamW classification layer routing
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Patchify
        x = x.unfold(2, p, p).unfold(3, p, p)  # Shape: (B, C, H/p, W/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)  # Shape: (B, num_patches, C*p*p)
        
        # Project patches
        x = self.patch_to_embed(x)
        
        # Add cls token and positional embedding
        cls_tokens = self.cls_token_embed.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # Transformer blocks
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.append(attn)
            
        # Classifier head
        x = self.norm(x[:, 0])
        return self.head(x), attn_maps

def compute_entropy(attn: torch.Tensor) -> float:
    """Compute average entropy of attention maps.
    
    attn shape: [B, heads, N, N]
    """
    entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1)
    return float(entropy.mean().item())

class DummyMNISTDataset(Dataset):
    """Dummy dataset for testing and fallback."""
    def __init__(self, num_images: int = 2000):
        self.num_images = num_images
        self.data = torch.randn(num_images, 1, 28, 28)
        self.targets = torch.randint(0, 10, (num_images,))

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]

def load_mnist_subset(num_images: int = 2000, batch_size: int = 64) -> DataLoader:
    """Load a subset of MNIST dataset with robust error fallback."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Cache directory path
    xdg_data_home = os.environ.get(
        "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
    )
    mnist_data_dir = os.path.join(xdg_data_home, "mnist_benchmark_data")
    
    try:
        train_dataset = datasets.MNIST(
            mnist_data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        indices = list(range(min(num_images, len(train_dataset))))
        subset = torch.utils.data.Subset(train_dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        logger.warning(f"Failed to load real MNIST dataset: {e}. Falling back to dummy random data.")
        dataset = DummyMNISTDataset(num_images)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def run_vit_experiment_logic(
    train_loader: DataLoader,
    max_steps: int = 60,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    model_kwargs: Optional[dict] = None,
    opt_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run ViT experiment logic, training models with various optimizers.
    
    Decoupled from plotting, file operations, and console outputs.
    """
    if model_kwargs is None:
        model_kwargs = dict(
            image_size=28,
            patch_size=4,
            num_classes=10,
            embed_dim=64,
            depth=2,
            heads=4
        )
    if opt_names is None:
        opt_names = ["AdamW", "Muon+Aux"]

    results = {}
    
    # Initialize templates under seed for identical start parameters
    torch.manual_seed(seed)
    np.random.seed(seed)
    template_model = SimpleViT(**model_kwargs).to(device)
    template_state = {k: v.clone() for k, v in template_model.state_dict().items()}

    for opt_name in opt_names:
        # Load fresh model copy with exactly matching template parameters
        model = SimpleViT(**model_kwargs).to(device)
        model.load_state_dict(template_state)

        # Set up optimizer
        if opt_name == "AdamW":
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        elif opt_name == "Muon+Aux":
            groups = create_muon_param_groups(
                model,
                muon_lr=0.02,
                adam_lr=3e-4,
                weight_decay=0.01
            )
            opt = SingleDeviceMuonWithAuxAdam(groups)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        losses = []
        accuracies = []
        entropies = []
        sv_snapshots = {}

        # Reset seed before each optimizer training run for deterministic shuffling/loading
        torch.manual_seed(seed)
        np.random.seed(seed)

        step = 0
        done = False
        while not done:
            for data, target in train_loader:
                if step >= max_steps:
                    done = True
                    break
                
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                output, attns = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                opt.step()

                # Calculate stats
                pred = output.argmax(dim=1)
                acc = (pred == target).float().mean().item()

                losses.append(float(loss.item()))
                accuracies.append(acc)
                entropies.append(compute_entropy(attns[0]))

                # Record Q projection singular values in layer 1 (index 0)
                with torch.no_grad():
                    qkv_weight = model.blocks[0].attn.qkv.weight
                    dim = model_kwargs["embed_dim"]
                    q_weight = qkv_weight[:dim]
                    s_vals = torch.linalg.svdvals(q_weight)
                    
                    if step in [0, 10, 30, 50] or step == max_steps - 1:
                        sv_snapshots[str(step)] = s_vals.tolist()

                step += 1
                
        results[opt_name] = {
            "losses": losses,
            "accuracies": accuracies,
            "entropies": entropies,
            "singular_values": sv_snapshots
        }

    return results

def run_vit_experiment() -> None:
    """Run standard experiment, write results, and log output."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    device = torch.device("cpu")
    logger.info("Loading MNIST subset...")
    train_loader = load_mnist_subset(num_images=2000, batch_size=64)
    
    logger.info("Running ViT optimization experiments (60 steps on CPU)...")
    results = run_vit_experiment_logic(
        train_loader=train_loader,
        max_steps=60,
        device=device,
        seed=42
    )
    
    # Save JSON results
    REPO_ROOT = Path(__file__).resolve().parent
    out_dir = REPO_ROOT / "explorer" / "public" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "transformer_spectral.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"JSON results saved to {out_dir / 'transformer_spectral.json'}")

    # Generate and save plot
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Training Loss
    plt.subplot(1, 2, 1)
    for name, res in results.items():
        plt.plot(res["losses"], label=name)
    plt.title("ViT MNIST Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Q-projection Singular Values at Step 50
    plt.subplot(1, 2, 2)
    for name, res in results.items():
        # Look for the last captured step in the singular values dict (usually '59')
        last_step = max(res["singular_values"].keys(), key=int)
        plt.plot(res["singular_values"][last_step], label=f"{name} (step {last_step})")
    plt.title("Q-Projection Singular Values")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    assets_dir = REPO_ROOT / "assets"
    assets_dir.mkdir(exist_ok=True)
    plot_path = assets_dir / "transformer_spectral_benchmark.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Plot saved to {plot_path}")

    # Rich summary table
    console = Console()
    table = Table(title="Vision Transformer Optimizer Comparison (MNIST)")
    table.add_column("Optimizer", style="cyan")
    table.add_column("Final Loss", style="green")
    table.add_column("Final Train Accuracy", style="yellow")
    table.add_column("Final Attention Entropy", style="magenta")
    
    for name, res in results.items():
        table.add_row(
            name,
            f"{res['losses'][-1]:.4f}",
            f"{res['accuracies'][-1]*100:.2f}%",
            f"{res['entropies'][-1]:.4f}"
        )
    console.print(table)

if __name__ == "__main__":
    run_vit_experiment()
