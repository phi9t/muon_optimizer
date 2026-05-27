import torch
import pytest
from torch.utils.data import DataLoader
from transformer_spectral_benchmark import (
    SimpleViT,
    DummyMNISTDataset,
    run_vit_experiment_logic,
)

def test_simple_vit_shape():
    # Model configuration
    # 28x28 MNIST images, patch_size=4, embed_dim=64, depth=2, heads=4, 10 classes
    model = SimpleViT(
        image_size=28,
        patch_size=4,
        num_classes=10,
        embed_dim=64,
        depth=2,
        heads=4
    )
    
    # Dummy input representing batch of 4 MNIST images
    # Shape: (B, C, H, W) = (4, 1, 28, 28)
    x = torch.randn(4, 1, 28, 28)
    
    logits, attn_maps = model(x)
    
    # Check output shapes
    assert logits.shape == (4, 10)
    assert len(attn_maps) == 2  # depth=2
    # Check attention map shape: (B, heads, N, N) where N = num_patches + 1
    # num_patches = (28 // 4) ** 2 = 7 * 7 = 49. N = 50.
    for attn in attn_maps:
        assert attn.shape == (4, 4, 50, 50)

def test_run_vit_experiment_logic():
    # Set up a small dummy dataloader
    dataset = DummyMNISTDataset(num_images=16)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model_kwargs = dict(
        image_size=28,
        patch_size=4,
        num_classes=10,
        embed_dim=16,  # small embed dim for fast test
        depth=1,       # shallow depth
        heads=2
    )
    
    max_steps = 2
    
    results = run_vit_experiment_logic(
        train_loader=train_loader,
        max_steps=max_steps,
        device=torch.device("cpu"),
        seed=42,
        model_kwargs=model_kwargs,
        opt_names=["AdamW", "Muon+Aux"]
    )
    
    assert isinstance(results, dict)
    for opt_name in ["AdamW", "Muon+Aux"]:
        assert opt_name in results
        res = results[opt_name]
        
        # Check metrics are tracked
        assert "losses" in res
        assert "accuracies" in res
        assert "entropies" in res
        assert "singular_values" in res
        
        assert len(res["losses"]) == max_steps
        assert len(res["accuracies"]) == max_steps
        assert len(res["entropies"]) == max_steps
        
        # At step 0 and step 1 (max_steps - 1), singular values should be recorded
        assert "0" in res["singular_values"]
        assert "1" in res["singular_values"]
        assert len(res["singular_values"]["0"]) == model_kwargs["embed_dim"]
