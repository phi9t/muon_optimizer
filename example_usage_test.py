"""
Unit tests for example_usage.py module.

This test suite verifies that all the examples in example_usage.py work correctly
and that the various Muon optimizer configurations produce expected results.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from unittest.mock import patch, MagicMock

from example_usage import (
    SimpleNet,
    ConvNet,
    create_dummy_data,
    train_model,
    example_1_basic_muon,
    example_2_hybrid_optimization,
    example_3_manual_parameter_grouping,
    example_4_convolutional_network,
    example_5_learning_rate_scheduling,
    example_6_parameter_analysis,
    run_all_examples,
    create_training_results_table,
    create_model_summary_table,
    create_optimizer_config_table,
)
from muon_optimizer import (
    SingleDeviceMuon,
    SingleDeviceMuonWithAuxAdam,
    create_muon_param_groups,
)

# Set up test logging (quieter than the examples)
logging.basicConfig(level=logging.CRITICAL)  # Suppress logs during testing


class TestModels:
    """Test the model definitions."""
    
    def test_simple_net_creation(self):
        """Test SimpleNet can be created and has correct structure."""
        model = SimpleNet(input_size=784, hidden_size=512, num_classes=10)
        
        # Check architecture
        assert isinstance(model.fc1, nn.Linear)
        assert isinstance(model.fc2, nn.Linear)
        assert isinstance(model.fc3, nn.Linear)
        assert isinstance(model.dropout, nn.Dropout)
        
        # Check dimensions
        assert model.fc1.in_features == 784
        assert model.fc1.out_features == 512
        assert model.fc3.out_features == 10
    
    def test_simple_net_forward(self):
        """Test SimpleNet forward pass."""
        model = SimpleNet()
        x = torch.randn(4, 784)
        output = model(x)
        
        assert output.shape == (4, 10)
        assert torch.all(torch.isfinite(output))
    
    def test_conv_net_creation(self):
        """Test ConvNet can be created and has correct structure."""
        model = ConvNet(num_classes=10)
        
        # Check architecture
        assert isinstance(model.conv1, nn.Conv2d)
        assert isinstance(model.conv2, nn.Conv2d)
        assert isinstance(model.fc1, nn.Linear)
        assert isinstance(model.fc2, nn.Linear)
        
        # Check dimensions
        assert model.conv1.in_channels == 1
        assert model.conv1.out_channels == 32
        assert model.fc2.out_features == 10
    
    def test_conv_net_forward(self):
        """Test ConvNet forward pass."""
        model = ConvNet()
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (4, 10)
        assert torch.all(torch.isfinite(output))


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_dummy_data(self):
        """Test dummy data creation."""
        X, y = create_dummy_data(num_samples=100, input_size=50, num_classes=5)
        
        assert X.shape == (100, 50)
        assert y.shape == (100,)
        assert torch.all(y >= 0)
        assert torch.all(y < 5)
        assert torch.all(torch.isfinite(X))
    
    def test_create_dummy_data_reproducible(self):
        """Test that dummy data creation is reproducible."""
        X1, y1 = create_dummy_data(num_samples=50)
        X2, y2 = create_dummy_data(num_samples=50)
        
        # Should be identical due to fixed seed
        assert torch.allclose(X1, X2)
        assert torch.equal(y1, y2)
    
    def test_train_model_basic(self):
        """Test basic model training functionality."""
        model = SimpleNet(input_size=10, hidden_size=20, num_classes=3)
        X, y = create_dummy_data(num_samples=32, input_size=10, num_classes=3)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16)
        
        # Use a simple SGD optimizer for testing
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Train for 1 epoch (just to test the function works)
        initial_params = [p.clone() for p in model.parameters()]
        trained_model, metrics = train_model(model, optimizer, dataloader, num_epochs=1)
        final_params = [p.clone() for p in trained_model.parameters()]
        
        # Parameters should have changed
        for initial, final in zip(initial_params, final_params):
            assert not torch.allclose(initial, final, atol=1e-6)
        
        # Should return the same model instance and metrics
        assert trained_model is model
        assert isinstance(metrics, list)
        assert len(metrics) == 1  # 1 epoch
        assert 'epoch' in metrics[0]
        assert 'loss' in metrics[0]
        assert 'accuracy' in metrics[0]
        assert 'time' in metrics[0]


class TestTableFunctions:
    """Test the rich table creation functions."""
    
    @patch('example_usage.rich_console')
    def test_create_training_results_table(self, mock_console):
        """Test training results table creation."""
        metrics = [
            {'epoch': 1, 'loss': 2.3, 'accuracy': 10.5, 'time': 1.2},
            {'epoch': 2, 'loss': 1.8, 'accuracy': 45.2, 'time': 1.1},
        ]
        
        # Should not raise an exception
        create_training_results_table(metrics, "Test Table")
        
        # Should call console.print once
        assert mock_console.return_value.print.call_count == 1
    
    @patch('example_usage.rich_console')
    def test_create_model_summary_table(self, mock_console):
        """Test model summary table creation."""
        model = SimpleNet(input_size=10, hidden_size=5, num_classes=2)
        
        # Should not raise an exception
        create_model_summary_table(model, "TestOptimizer")
        
        # Should call console.print once
        assert mock_console.return_value.print.call_count == 1
    
    @patch('example_usage.rich_console')
    def test_create_optimizer_config_table(self, mock_console):
        """Test optimizer config table creation."""
        model = SimpleNet(input_size=10, hidden_size=5, num_classes=2)
        optimizer = SingleDeviceMuon(model.parameters(), lr=0.01)
        
        # Should not raise an exception
        create_optimizer_config_table(optimizer)
        
        # Should call console.print once
        assert mock_console.return_value.print.call_count == 1


class TestOptimizationExamples:
    """Test the optimization examples."""
    
    @patch('example_usage.rich_console')
    @patch('example_usage.log')
    def test_example_1_basic_muon(self, mock_log, mock_console):
        """Test example 1 - basic Muon usage."""
        model, optimizer, metrics = example_1_basic_muon()
        
        # Check types
        assert isinstance(model, SimpleNet)
        assert isinstance(optimizer, SingleDeviceMuon)
        assert isinstance(metrics, list)
        assert len(metrics) == 3  # 3 epochs
        
        # Check optimizer configuration
        assert optimizer.param_groups[0]['lr'] == 0.02
        assert optimizer.param_groups[0]['momentum'] == 0.95
        assert optimizer.param_groups[0]['weight_decay'] == 0.01
    
    @patch('example_usage.rich_console')
    @patch('example_usage.log')
    def test_example_2_hybrid_optimization(self, mock_log, mock_console):
        """Test example 2 - hybrid Muon/Adam optimization."""
        model, optimizer, metrics = example_2_hybrid_optimization()
        
        # Check types
        assert isinstance(model, SimpleNet)
        assert isinstance(optimizer, SingleDeviceMuonWithAuxAdam)
        assert isinstance(metrics, list)
        
        # Should have both Muon and AdamW parameter groups
        muon_groups = [g for g in optimizer.param_groups if g["use_muon"]]
        adam_groups = [g for g in optimizer.param_groups if not g["use_muon"]]
        
        assert len(muon_groups) > 0
        assert len(adam_groups) > 0
    
    @patch('example_usage.rich_console')
    @patch('example_usage.log')
    def test_example_3_manual_parameter_grouping(self, mock_log, mock_console):
        """Test example 3 - manual parameter grouping."""
        model, optimizer, metrics = example_3_manual_parameter_grouping()
        
        # Check types
        assert isinstance(model, SimpleNet)
        assert isinstance(optimizer, SingleDeviceMuonWithAuxAdam)
        assert isinstance(metrics, list)
        
        # Should have exactly 2 groups (matrix and scalar)
        assert len(optimizer.param_groups) == 2
        
        # Check that groups have the expected configuration
        muon_group = next(g for g in optimizer.param_groups if g["use_muon"])
        adam_group = next(g for g in optimizer.param_groups if not g["use_muon"])
        
        assert muon_group["lr"] == 0.02
        assert adam_group["lr"] == 3e-4
    
    @patch('example_usage.rich_console')
    @patch('example_usage.log')
    def test_example_4_convolutional_network(self, mock_log, mock_console):
        """Test example 4 - convolutional network."""
        model, optimizer, metrics = example_4_convolutional_network()
        
        # Check types
        assert isinstance(model, ConvNet)
        assert isinstance(optimizer, SingleDeviceMuon)
        assert isinstance(metrics, list)
        
        # Check that model has convolutional layers
        conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        assert len(conv_layers) > 0
    
    @patch('example_usage.rich_console')
    @patch('example_usage.log')
    def test_example_5_learning_rate_scheduling(self, mock_log, mock_console):
        """Test example 5 - learning rate scheduling."""
        model, optimizer, metrics = example_5_learning_rate_scheduling()
        
        # Check types
        assert isinstance(model, SimpleNet)
        assert isinstance(optimizer, SingleDeviceMuon)
        assert isinstance(metrics, list)
        
        # Learning rate should have been reduced by the scheduler
        final_lr = optimizer.param_groups[0]['lr']
        initial_lr = 0.02
        
        # After 3 steps with gamma=0.9: lr = 0.02 * 0.9^3 ≈ 0.01458
        expected_lr = initial_lr * (0.9 ** 3)
        assert abs(final_lr - expected_lr) < 1e-6
        
        # Metrics should have LR information
        for metric in metrics:
            assert 'lr' in metric
    
    @patch('example_usage.rich_console')
    @patch('example_usage.log')
    def test_example_6_parameter_analysis(self, mock_log, mock_console):
        """Test example 6 - parameter analysis."""
        models = example_6_parameter_analysis()
        
        assert isinstance(models, dict)
        assert "SimpleNet (MLP)" in models
        assert "ConvNet (CNN)" in models
        
        assert isinstance(models["SimpleNet (MLP)"], SimpleNet)
        assert isinstance(models["ConvNet (CNN)"], ConvNet)


class TestParameterGrouping:
    """Test parameter grouping logic."""
    
    def test_create_muon_param_groups_simple_net(self):
        """Test parameter grouping for SimpleNet."""
        model = SimpleNet()
        param_groups = create_muon_param_groups(model)
        
        assert len(param_groups) == 2  # Muon group and Adam group
        
        muon_group = next(g for g in param_groups if g["use_muon"])
        adam_group = next(g for g in param_groups if not g["use_muon"])
        
        # Check Muon group contains 2D+ parameters
        for param in muon_group["params"]:
            assert param.ndim >= 2
        
        # Check Adam group contains 1D parameters
        for param in adam_group["params"]:
            assert param.ndim == 1
    
    def test_create_muon_param_groups_conv_net(self):
        """Test parameter grouping for ConvNet."""
        model = ConvNet()
        param_groups = create_muon_param_groups(model)
        
        assert len(param_groups) == 2
        
        muon_group = next(g for g in param_groups if g["use_muon"])
        adam_group = next(g for g in param_groups if not g["use_muon"])
        
        # ConvNet should have conv weights and linear weights in Muon group
        muon_param_count = len(muon_group["params"])
        adam_param_count = len(adam_group["params"])
        
        assert muon_param_count > 0
        assert adam_param_count > 0  # Biases should go to Adam group
    
    def test_manual_parameter_separation(self):
        """Test manual parameter separation logic."""
        model = SimpleNet()
        
        matrix_params = []
        scalar_params = []
        
        for param in model.parameters():
            if param.ndim >= 2:
                matrix_params.append(param)
            else:
                scalar_params.append(param)
        
        # SimpleNet has 3 linear layers, so 3 weight matrices and 3 biases
        assert len(matrix_params) == 3  # fc1.weight, fc2.weight, fc3.weight
        assert len(scalar_params) == 3   # fc1.bias, fc2.bias, fc3.bias
        
        # Check dimensions
        for param in matrix_params:
            assert param.ndim == 2
        for param in scalar_params:
            assert param.ndim == 1


class TestIntegration:
    """Integration tests."""
    
    @patch('example_usage.rich_console')
    @patch('example_usage.log')
    def test_run_all_examples_mock(self, mock_log, mock_console):
        """Test that all examples can run successfully with mocked output."""
        # Mock the rich console to avoid actual table printing during tests
        mock_console.return_value.print = MagicMock()
        
        results = run_all_examples()
        
        # Check that we got results for all examples
        expected_keys = ['basic', 'hybrid', 'manual', 'conv', 'scheduler', 'analysis']
        for key in expected_keys:
            assert key in results
        
        # Check types of results
        assert isinstance(results['basic'], tuple)
        assert isinstance(results['hybrid'], tuple)
        assert isinstance(results['manual'], tuple)
        assert isinstance(results['conv'], tuple)
        assert isinstance(results['scheduler'], tuple)
        assert isinstance(results['analysis'], dict)
        
        # Check that models and optimizers are correct types
        for key in ['basic', 'hybrid', 'manual', 'conv', 'scheduler']:
            model, optimizer, metrics = results[key]
            assert isinstance(model, (SimpleNet, ConvNet))
            assert isinstance(optimizer, (SingleDeviceMuon, SingleDeviceMuonWithAuxAdam))
            assert isinstance(metrics, list)
            assert len(metrics) > 0  # Should have training metrics
    
    def test_model_parameter_counts(self):
        """Test that models have expected parameter counts."""
        simple_net = SimpleNet(input_size=784, hidden_size=512, num_classes=10)
        conv_net = ConvNet(num_classes=10)
        
        simple_count = sum(p.numel() for p in simple_net.parameters())
        conv_count = sum(p.numel() for p in conv_net.parameters())
        
        # SimpleNet: (784*512 + 512) + (512*512 + 512) + (512*10 + 10)
        expected_simple = 784*512 + 512 + 512*512 + 512 + 512*10 + 10
        assert simple_count == expected_simple
        
        # ConvNet has conv layers + linear layers
        assert conv_count > 0  # Just check it's not zero
    
    def test_optimizer_state_initialization(self):
        """Test that optimizers initialize state correctly."""
        model = SimpleNet(input_size=10, hidden_size=20, num_classes=2)
        
        # Test SingleDeviceMuon
        muon_optimizer = SingleDeviceMuon(model.parameters(), lr=0.01)
        
        # Initially no state
        assert len(muon_optimizer.state) == 0
        
        # After one step, should have state
        X, y = create_dummy_data(num_samples=8, input_size=10, num_classes=2)
        output = model(X)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        muon_optimizer.step()
        
        # Should now have momentum buffers
        assert len(muon_optimizer.state) > 0
        
        for param in model.parameters():
            if param in muon_optimizer.state:
                state = muon_optimizer.state[param]
                assert "momentum_buffer" in state
                assert state["momentum_buffer"].shape == param.shape


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_model(self):
        """Test behavior with a model that has no parameters."""
        class EmptyModel(nn.Module):
            def forward(self, x):
                return x
        
        model = EmptyModel()
        
        # Should not crash when creating param groups
        param_groups = create_muon_param_groups(model)
        
        # Both groups should be empty
        muon_group = next(g for g in param_groups if g["use_muon"])
        adam_group = next(g for g in param_groups if not g["use_muon"])
        
        assert len(muon_group["params"]) == 0
        assert len(adam_group["params"]) == 0
    
    def test_small_batch_training(self):
        """Test training with very small batches."""
        model = SimpleNet(input_size=5, hidden_size=8, num_classes=2)
        X, y = create_dummy_data(num_samples=4, input_size=5, num_classes=2)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=1)  # Very small batch
        
        optimizer = SingleDeviceMuon(model.parameters(), lr=0.01)
        
        # Should not crash with batch size 1
        trained_model, metrics = train_model(model, optimizer, dataloader, num_epochs=1)
        assert trained_model is model
        assert isinstance(metrics, list)
        assert len(metrics) == 1
    
    def test_zero_learning_rate(self):
        """Test optimizer with zero learning rate."""
        model = SimpleNet(input_size=5, hidden_size=8, num_classes=2)
        optimizer = SingleDeviceMuon(model.parameters(), lr=0.0)
        
        X, y = create_dummy_data(num_samples=4, input_size=5, num_classes=2)
        
        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Forward and backward pass
        output = model(X)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
        
        # Parameters should not change with lr=0
        for initial, current in zip(initial_params, model.parameters()):
            assert torch.allclose(initial, current, atol=1e-8)


class TestMetricsStructure:
    """Test the structure of training metrics returned by functions."""
    
    def test_training_metrics_structure(self):
        """Test that training metrics have the expected structure."""
        model = SimpleNet(input_size=10, hidden_size=5, num_classes=2)
        X, y = create_dummy_data(num_samples=16, input_size=10, num_classes=2)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=8)
        
        optimizer = SingleDeviceMuon(model.parameters(), lr=0.01)
        
        trained_model, metrics = train_model(model, optimizer, dataloader, num_epochs=2)
        
        assert len(metrics) == 2  # 2 epochs
        
        for i, metric in enumerate(metrics):
            assert isinstance(metric, dict)
            assert metric['epoch'] == i + 1
            assert isinstance(metric['loss'], float)
            assert isinstance(metric['accuracy'], float)
            assert isinstance(metric['time'], float)
            assert metric['accuracy'] >= 0.0
            assert metric['accuracy'] <= 100.0
            assert metric['time'] > 0.0
    
    @patch('example_usage.rich_console')
    @patch('example_usage.log')
    def test_scheduler_metrics_structure(self, mock_log, mock_console):
        """Test that scheduler example returns metrics with LR info."""
        model, optimizer, metrics = example_5_learning_rate_scheduling()
        
        assert len(metrics) == 3  # 3 epochs
        
        for metric in metrics:
            assert 'lr' in metric
            assert isinstance(metric['lr'], float)
            assert metric['lr'] > 0.0


if __name__ == "__main__":
    pytest.main([__file__])