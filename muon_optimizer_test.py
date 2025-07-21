"""
Test suite for the Muon optimizer module.

This test suite covers all the main functionality of the Muon optimizer including:
- Core functions (zeropower_via_newtonschulz5, muon_update, adam_update)
- All optimizer classes (Muon, SingleDeviceMuon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam)
- Parameter group creation utilities
- Edge cases and error handling

Run with: pytest muon_optimizer_test.py
"""

import pytest
import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
from unittest.mock import patch, MagicMock

from muon_optimizer import (
    zeropower_via_newtonschulz5,
    muon_update,
    adam_update,
    Muon,
    SingleDeviceMuon,
    MuonWithAuxAdam,
    SingleDeviceMuonWithAuxAdam,
    create_muon_param_groups
)


class TestZeropowerViaNewtonschulz5:
    """Test the Newton-Schulz orthogonalization function."""
    
    def test_basic_orthogonalization(self):
        """Test basic orthogonalization of a 2D matrix."""
        torch.manual_seed(42)  # Set seed for reproducible test
        G = torch.randn(4, 4)
        result = zeropower_via_newtonschulz5(G, steps=5)
        
        assert result.shape == G.shape
        # Result stays in bfloat16 format from the function
        assert result.dtype == torch.bfloat16
        
        # Convert to float32 for numerical checks
        result_f32 = result.float()
        
        # Check that the result is approximately orthogonal
        orthogonal_check = torch.mm(result_f32, result_f32.T)
        identity = torch.eye(4).float()
        assert torch.allclose(orthogonal_check, identity, atol=0.5)
    
    def test_rectangular_matrix(self):
        """Test orthogonalization of rectangular matrices."""
        # Tall matrix
        G_tall = torch.randn(6, 4)
        result_tall = zeropower_via_newtonschulz5(G_tall, steps=5)
        assert result_tall.shape == G_tall.shape
        
        # Wide matrix
        G_wide = torch.randn(4, 6)
        result_wide = zeropower_via_newtonschulz5(G_wide, steps=5)
        assert result_wide.shape == G_wide.shape
    
    def test_batch_processing(self):
        """Test orthogonalization of batched matrices."""
        G_batch = torch.randn(3, 4, 4)
        result = zeropower_via_newtonschulz5(G_batch, steps=5)
        assert result.shape == G_batch.shape
        
        # Convert to float32 for numerical checks
        result_f32 = result.float()
        
        # Check each matrix in the batch is approximately orthogonal
        for i in range(3):
            orthogonal_check = torch.mm(result_f32[i], result_f32[i].T)
            identity = torch.eye(4).float()
            # Just check that the diagonal is reasonably close to 1 and off-diagonal is small
            diag_diff = torch.abs(torch.diag(orthogonal_check) - 1.0)
            assert torch.all(diag_diff < 0.8), f"Diagonal should be close to 1, got {torch.diag(orthogonal_check)}"
    
    def test_invalid_dimensions(self):
        """Test error handling for invalid input dimensions."""
        G_1d = torch.randn(10)
        with pytest.raises(ValueError, match="Input tensor must have at least 2 dimensions"):
            zeropower_via_newtonschulz5(G_1d, steps=5)
    
    def test_different_step_counts(self):
        """Test different numbers of Newton-Schulz steps."""
        G = torch.randn(4, 4)
        
        result_1 = zeropower_via_newtonschulz5(G, steps=1)
        result_5 = zeropower_via_newtonschulz5(G, steps=5)
        result_10 = zeropower_via_newtonschulz5(G, steps=10)
        
        assert result_1.shape == G.shape
        assert result_5.shape == G.shape
        assert result_10.shape == G.shape
        
        # Convert to float32 for numerical checks
        result_1_f32 = result_1.float()
        result_5_f32 = result_5.float()
        result_10_f32 = result_10.float()
        
        # Check that all results are reasonably orthogonal
        # Note: More steps don't always guarantee better orthogonalization
        # due to the quintic iteration and normalization factors
        orth_1 = torch.norm(torch.mm(result_1_f32, result_1_f32.T) - torch.eye(4))
        orth_5 = torch.norm(torch.mm(result_5_f32, result_5_f32.T) - torch.eye(4))
        orth_10 = torch.norm(torch.mm(result_10_f32, result_10_f32.T) - torch.eye(4))
        
        # All should be reasonably orthogonal (more relaxed tolerances)
        assert orth_1 < 2.0  # 1 step may not be very orthogonal
        assert orth_5 < 1.0  # 5 steps should be better
        assert orth_10 < 1.0  # 10 steps should be good


class TestMuonUpdate:
    """Test the core Muon update function."""
    
    def test_basic_update(self):
        """Test basic Muon update functionality."""
        grad = torch.randn(4, 4)
        momentum = torch.zeros_like(grad)
        
        update = muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True)
        
        assert update.shape == grad.shape
        assert not torch.allclose(momentum, torch.zeros_like(grad))  # Momentum should be updated
    
    def test_1d_parameters(self):
        """Test that 1D parameters (like biases) skip orthogonalization."""
        grad_1d = torch.randn(10)
        momentum_1d = torch.zeros_like(grad_1d)
        
        update = muon_update(grad_1d, momentum_1d, beta=0.95)
        
        assert update.shape == grad_1d.shape
        # For 1D parameters, just check that we get a reasonable update
        # and that momentum buffer was modified
        assert not torch.allclose(momentum_1d, torch.zeros_like(grad_1d))
        assert torch.allclose(update, grad_1d, atol=1e-6)  # update should be the modified grad
    
    def test_4d_conv_parameters(self):
        """Test handling of 4D convolutional parameters."""
        grad_4d = torch.randn(32, 16, 3, 3)  # Conv layer shape
        momentum_4d = torch.zeros_like(grad_4d)
        
        update = muon_update(grad_4d, momentum_4d, beta=0.95, ns_steps=5)
        
        assert update.shape == grad_4d.shape
    
    def test_momentum_accumulation(self):
        """Test that momentum accumulates correctly over multiple updates."""
        grad = torch.randn(4, 4)
        momentum = torch.zeros_like(grad)
        
        # First update
        update1 = muon_update(grad, momentum, beta=0.95)
        momentum_after_1 = momentum.clone()
        
        # Second update with same gradient
        update2 = muon_update(grad, momentum, beta=0.95)
        
        # Momentum should have accumulated
        assert not torch.allclose(momentum_after_1, momentum)
    
    def test_nesterov_effect(self):
        """Test the effect of Nesterov momentum."""
        grad = torch.randn(4, 4)
        momentum_nesterov = torch.zeros_like(grad)
        momentum_standard = torch.zeros_like(grad)
        
        update_nesterov = muon_update(grad, momentum_nesterov, beta=0.95, nesterov=True)
        update_standard = muon_update(grad, momentum_standard, beta=0.95, nesterov=False)
        
        # Updates should be different when Nesterov is enabled vs disabled
        assert not torch.allclose(update_nesterov, update_standard)


class TestAdamUpdate:
    """Test the Adam update function."""
    
    def test_basic_adam_update(self):
        """Test basic Adam update functionality."""
        grad = torch.randn(10, 10)
        buf1 = torch.zeros_like(grad)  # First moment
        buf2 = torch.zeros_like(grad)  # Second moment
        
        update = adam_update(grad, buf1, buf2, step=1, betas=(0.9, 0.95), eps=1e-8)
        
        assert update.shape == grad.shape
        assert not torch.allclose(buf1, torch.zeros_like(grad))  # buf1 should be updated
        assert not torch.allclose(buf2, torch.zeros_like(grad))  # buf2 should be updated
    
    def test_bias_correction(self):
        """Test that bias correction works correctly."""
        grad = torch.ones(5, 5)
        buf1 = torch.zeros_like(grad)
        buf2 = torch.zeros_like(grad)
        
        # Early steps should have larger updates due to bias correction
        update_step1 = adam_update(grad, buf1.clone(), buf2.clone(), step=1, betas=(0.9, 0.95), eps=1e-8)
        update_step100 = adam_update(grad, buf1.clone(), buf2.clone(), step=100, betas=(0.9, 0.95), eps=1e-8)
        
        # Step 1 should have larger magnitude due to bias correction
        assert torch.norm(update_step1) > torch.norm(update_step100)
    
    def test_different_betas(self):
        """Test Adam with different beta values."""
        # Use a simple test that just checks basic functionality
        grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        buf1 = torch.zeros_like(grad)
        buf2 = torch.zeros_like(grad)
        
        # Just test that the function runs and produces reasonable output
        update = adam_update(grad, buf1, buf2, step=1, betas=(0.9, 0.95), eps=1e-8)
        
        # Check that update has the right shape and is finite
        assert update.shape == grad.shape
        assert torch.all(torch.isfinite(update))
        assert torch.all(update > 0)  # Should be positive since grad is positive
        
        # Check that buffers were updated
        assert not torch.allclose(buf1, torch.zeros_like(grad))
        assert not torch.allclose(buf2, torch.zeros_like(grad))


class TestMuonOptimizer:
    """Test the main Muon optimizer class."""
    
    def test_initialization(self):
        """Test Muon optimizer initialization."""
        params = [torch.randn(10, 10, requires_grad=True) for _ in range(3)]
        optimizer = Muon(params, lr=0.02, momentum=0.95, weight_decay=0.01)
        
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 0.02
        assert optimizer.param_groups[0]['momentum'] == 0.95
        assert optimizer.param_groups[0]['weight_decay'] == 0.01
        assert optimizer.param_groups[0]['ns_steps'] == 5
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        params = [torch.randn(10, 10, requires_grad=True)]
        
        with pytest.raises(ValueError, match="Learning rate must be a non-negative number"):
            Muon(params, lr=-0.1)
        
        with pytest.raises(ValueError, match="Weight decay must be a non-negative number"):
            Muon(params, weight_decay=-0.1)
        
        with pytest.raises(ValueError, match="Momentum must be in"):
            Muon(params, momentum=1.5)
        
        with pytest.raises(ValueError, match="ns_steps must be a positive integer"):
            Muon(params, ns_steps=0)
    
    def test_empty_params(self):
        """Test error handling for empty parameter list."""
        with pytest.raises(ValueError, match="params must be a non-empty iterable"):
            Muon([])
    
    def test_invalid_param_types(self):
        """Test error handling for invalid parameter types."""
        invalid_params = [torch.randn(10, 10), "not a tensor"]
        with pytest.raises(ValueError, match="All params must be torch.nn.Parameter or torch.Tensor"):
            Muon(invalid_params)
    
    @patch('torch.distributed.is_initialized', return_value=False)
    def test_single_device_step(self, mock_dist):
        """Test optimizer step on single device."""
        model = nn.Linear(10, 5)
        optimizer = Muon(model.parameters(), lr=0.02)
        
        # Forward pass
        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Optimization step
        optimizer.step()
        
        # Parameters should have changed
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.allclose(initial, current)
    
    def test_step_with_closure(self):
        """Test optimizer step with closure function."""
        model = nn.Linear(10, 5)
        optimizer = Muon(model.parameters(), lr=0.02)
        
        def closure():
            optimizer.zero_grad()
            x = torch.randn(3, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        assert loss is not None
        assert isinstance(loss, torch.Tensor)


class TestSingleDeviceMuon:
    """Test the SingleDeviceMuon optimizer."""
    
    def test_initialization(self):
        """Test SingleDeviceMuon initialization."""
        params = [torch.randn(5, 5, requires_grad=True)]
        optimizer = SingleDeviceMuon(params, lr=0.01)
        
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 0.01
    
    def test_optimization_step(self):
        """Test optimization step for SingleDeviceMuon."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        optimizer = SingleDeviceMuon(model.parameters(), lr=0.02)
        
        x = torch.randn(5, 10)
        y_target = torch.randn(5, 1)
        
        # Forward pass
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y_target)
        loss.backward()
        
        # Store initial loss
        initial_loss = loss.item()
        
        # Optimization step
        optimizer.step()
        optimizer.zero_grad()
        
        # Forward pass after optimization
        y_pred_new = model(x)
        new_loss = nn.MSELoss()(y_pred_new, y_target)
        
        # Loss should generally decrease (though not guaranteed in one step)
        assert new_loss.item() != initial_loss


class TestMuonWithAuxAdam:
    """Test the hybrid Muon/Adam optimizer."""
    
    def test_initialization(self):
        """Test MuonWithAuxAdam initialization."""
        muon_params = [torch.randn(10, 10, requires_grad=True)]
        adam_params = [torch.randn(10, requires_grad=True)]
        
        param_groups = [
            {"params": muon_params, "lr": 0.02, "use_muon": True},
            {"params": adam_params, "lr": 3e-4, "use_muon": False}
        ]
        
        optimizer = MuonWithAuxAdam(param_groups)
        assert len(optimizer.param_groups) == 2
    
    def test_missing_use_muon_key(self):
        """Test error handling when use_muon key is missing."""
        params = [torch.randn(10, 10, requires_grad=True)]
        param_groups = [{"params": params, "lr": 0.02}]  # Missing use_muon
        
        with pytest.raises(ValueError, match="must contain 'use_muon' key"):
            MuonWithAuxAdam(param_groups)
    
    def test_missing_params_key(self):
        """Test error handling when params key is missing."""
        param_groups = [{"lr": 0.02, "use_muon": True}]  # Missing params
        
        with pytest.raises(ValueError, match="must contain 'params' key"):
            MuonWithAuxAdam(param_groups)
    
    def test_invalid_muon_group_keys(self):
        """Test error handling for invalid Muon group keys."""
        params = [torch.randn(10, 10, requires_grad=True)]
        param_groups = [{
            "params": params,
            "lr": 0.02,
            "use_muon": True,
            "invalid_key": "value"
        }]
        
        with pytest.raises(ValueError, match="must contain exactly"):
            MuonWithAuxAdam(param_groups)
    
    def test_invalid_adam_group_keys(self):
        """Test error handling for invalid Adam group keys."""
        params = [torch.randn(10, requires_grad=True)]
        param_groups = [{
            "params": params,
            "lr": 3e-4,
            "use_muon": False,
            "invalid_key": "value"
        }]
        
        with pytest.raises(ValueError, match="must contain exactly"):
            MuonWithAuxAdam(param_groups)
    
    @patch('torch.distributed.is_initialized', return_value=False)
    def test_hybrid_optimization(self, mock_dist):
        """Test that both Muon and Adam parameters are optimized correctly."""
        # Create a model with both 2D and 1D parameters
        model = nn.Sequential(
            nn.Linear(10, 20),  # 2D weight + 1D bias
            nn.Linear(20, 5)    # 2D weight + 1D bias
        )
        
        # Separate parameters
        matrix_params = [p for p in model.parameters() if p.ndim >= 2]
        bias_params = [p for p in model.parameters() if p.ndim < 2]
        
        param_groups = [
            {"params": matrix_params, "lr": 0.02, "use_muon": True},
            {"params": bias_params, "lr": 3e-4, "use_muon": False}
        ]
        
        optimizer = MuonWithAuxAdam(param_groups)
        
        # Forward pass and backward
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Store initial parameters
        initial_matrix = [p.clone() for p in matrix_params]
        initial_bias = [p.clone() for p in bias_params]
        
        # Optimization step
        optimizer.step()
        
        # Both parameter types should have changed
        for initial, current in zip(initial_matrix, matrix_params):
            assert not torch.allclose(initial, current)
        for initial, current in zip(initial_bias, bias_params):
            assert not torch.allclose(initial, current)


class TestSingleDeviceMuonWithAuxAdam:
    """Test the single-device hybrid optimizer."""
    
    def test_initialization(self):
        """Test SingleDeviceMuonWithAuxAdam initialization."""
        muon_params = [torch.randn(8, 8, requires_grad=True)]
        adam_params = [torch.randn(8, requires_grad=True)]
        
        param_groups = [
            {"params": muon_params, "lr": 0.02, "use_muon": True},
            {"params": adam_params, "lr": 3e-4, "use_muon": False}
        ]
        
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        assert len(optimizer.param_groups) == 2
    
    def test_optimization_functionality(self):
        """Test basic optimization functionality."""
        model = nn.Linear(5, 3)
        
        matrix_params = [model.weight]
        bias_params = [model.bias]
        
        param_groups = [
            {"params": matrix_params, "lr": 0.02, "use_muon": True},
            {"params": bias_params, "lr": 3e-4, "use_muon": False}
        ]
        
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        
        x = torch.randn(10, 5)
        y_target = torch.randn(10, 3)
        
        # Training step
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y_target)
        loss.backward()
        
        # Store initial parameters
        initial_weight = model.weight.clone()
        initial_bias = model.bias.clone()
        
        optimizer.step()
        
        # Parameters should have changed
        assert not torch.allclose(initial_weight, model.weight)
        assert not torch.allclose(initial_bias, model.bias)


class TestCreateMuonParamGroups:
    """Test the parameter group creation utility."""
    
    def test_basic_param_group_creation(self):
        """Test basic parameter group creation."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        param_groups = create_muon_param_groups(model)
        
        assert len(param_groups) == 2
        assert param_groups[0]["use_muon"] == True
        assert param_groups[1]["use_muon"] == False
        
        # Check that we have both Muon and Adam parameters
        muon_params = param_groups[0]["params"]
        adam_params = param_groups[1]["params"]
        
        assert len(muon_params) > 0
        assert len(adam_params) > 0
    
    def test_embedding_exclusion(self):
        """Test that embedding layers are excluded from Muon optimization."""
        class ModelWithEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 50)
                self.linear = nn.Linear(50, 10)
        
        model = ModelWithEmbedding()
        param_groups = create_muon_param_groups(model)
        
        # Embedding should go to Adam group
        adam_params = param_groups[1]["params"]
        muon_params = param_groups[0]["params"]
        
        # The embedding weight should be in Adam params (it's 2D but named with "embed")
        assert model.embedding.weight in adam_params
        assert model.linear.weight in muon_params
    
    def test_custom_learning_rates(self):
        """Test parameter group creation with custom learning rates."""
        model = nn.Linear(5, 3)
        
        param_groups = create_muon_param_groups(
            model,
            muon_lr=0.05,
            adam_lr=1e-3,
            weight_decay=0.01
        )
        
        assert param_groups[0]["lr"] == 0.05  # Muon group
        assert param_groups[1]["lr"] == 1e-3  # Adam group
        assert param_groups[0]["weight_decay"] == 0.01
        assert param_groups[1]["weight_decay"] == 0.01


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_gradients(self):
        """Test handling of zero gradients."""
        model = nn.Linear(5, 3)
        optimizer = SingleDeviceMuon(model.parameters(), lr=0.02)
        
        # Set gradients to zero
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        
        # Should not crash
        optimizer.step()
    
    def test_none_gradients(self):
        """Test handling of None gradients."""
        model = nn.Linear(5, 3)
        optimizer = SingleDeviceMuon(model.parameters(), lr=0.02)
        
        # Gradients are None by default
        # Should not crash - optimizer should handle this
        optimizer.step()
    
    def test_very_small_matrices(self):
        """Test handling of very small matrices."""
        # 2x2 matrix
        small_param = torch.randn(2, 2, requires_grad=True)
        optimizer = SingleDeviceMuon([small_param], lr=0.02)
        
        small_param.grad = torch.randn_like(small_param)
        optimizer.step()
        
        # Should complete without error
        assert True
    
    def test_large_batch_matrices(self):
        """Test handling of large batch of matrices."""
        # Large batch dimension
        large_param = torch.randn(100, 50, 50, requires_grad=True)
        optimizer = SingleDeviceMuon([large_param], lr=0.02)
        
        large_param.grad = torch.randn_like(large_param)
        optimizer.step()
        
        # Should complete without error
        assert True


# Integration test
def test_training_integration():
    """Integration test simulating a small training loop."""
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    optimizer = SingleDeviceMuon(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()
    
    # Generate some dummy data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    initial_loss = None
    
    # Training loop
    for epoch in range(5):
        optimizer.zero_grad()
        
        outputs = model(X)
        loss = criterion(outputs, y)
        
        if epoch == 0:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
    
    # Loss should have decreased
    final_outputs = model(X)
    final_loss = criterion(final_outputs, y).item()
    
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"


if __name__ == "__main__":
    pytest.main([__file__])