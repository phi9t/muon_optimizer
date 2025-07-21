"""
Example usage of the Muon optimizer.

This script demonstrates various ways to use the Muon optimizer including:
- Basic usage with SingleDeviceMuon
- Hybrid optimization with MuonWithAuxAdam
- Automatic parameter grouping
- Training a simple neural network
- Using different model architectures (MLP, CNN)
"""

import functools
import logging
import time
from typing import Dict, List

import rich.traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from torch.utils.data import DataLoader, TensorDataset

from muon_optimizer import (
    SingleDeviceMuon,
    SingleDeviceMuonWithAuxAdam,
    create_muon_param_groups,
)


class SimpleNet(nn.Module):
    """Simple neural network for demonstration."""

    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ConvNet(nn.Module):
    """Convolutional neural network for demonstration."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def create_dummy_data(num_samples=1000, input_size=784, num_classes=10):
    """Create dummy data for training."""
    torch.manual_seed(42)
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def train_model(model, optimizer, dataloader, num_epochs=5, device="cpu"):
    """Train a model with the given optimizer and return metrics."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    training_metrics = []

    log().info(f"[yellow]Training for {num_epochs} epochs on {device}[/yellow]")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        accuracy = 100.0 * correct / total

        training_metrics.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "accuracy": accuracy,
                "time": epoch_time,
            }
        )

        if epoch % 1 == 0:  # Log every epoch for demos
            log().info(
                f"[dim]Epoch {epoch+1:2d}[/dim] | "
                f"[red]Loss[/red]: {avg_loss:.4f} | "
                f"[green]Accuracy[/green]: {accuracy:.2f}% | "
                f"[yellow]Time[/yellow]: {epoch_time:.2f}s"
            )

    return model, training_metrics


def create_training_results_table(metrics: List[Dict], title: str) -> None:
    """Create a table showing training results."""
    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Epoch", style="dim", width=8)
    table.add_column("Loss", style="red", width=12)
    table.add_column("Accuracy", style="green", width=12)
    table.add_column("Time (s)", style="yellow", width=10)
    table.add_column("Progress", style="cyan", width=12)

    for metric in metrics:
        # Create a simple progress indicator
        accuracy = metric["accuracy"]
        progress = "█" * int(accuracy // 10) + "░" * (10 - int(accuracy // 10))

        table.add_row(
            str(metric["epoch"]),
            f"{metric['loss']:.4f}",
            f"{metric['accuracy']:.2f}%",
            f"{metric['time']:.2f}",
            f"[cyan]{progress}[/cyan]",
        )

    rich_console().print(table)


def create_model_summary_table(model, optimizer_name: str) -> None:
    """Create a table showing model architecture summary."""
    table = Table(
        title=f"Model Architecture Summary ({optimizer_name})",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Parameter", style="bold", width=20)
    table.add_column("Shape", style="cyan", width=18)
    table.add_column("Elements", style="yellow", width=12)
    table.add_column("Type", style="green", width=12)
    table.add_column("Optimizer", style="magenta", width=12)

    total_params = 0
    muon_params = 0
    adam_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        param_type = "Matrix" if param.ndim >= 2 else "Vector"
        opt_type = "Muon" if param.ndim >= 2 else "AdamW"

        if param.ndim >= 2:
            muon_params += param.numel()
        else:
            adam_params += param.numel()

        table.add_row(
            name, str(tuple(param.shape)), f"{param.numel():,}", param_type, opt_type
        )

    # Add summary row
    table.add_section()
    table.add_row(
        "[bold]Total",
        "",
        f"[bold]{total_params:,}",
        "",
        f"[bold]M:{muon_params:,} A:{adam_params:,}",
    )

    rich_console().print(table)


def create_optimizer_config_table(optimizer) -> None:
    """Create a table showing optimizer configuration."""
    table = Table(
        title="Optimizer Configuration",
        show_header=True,
        header_style="bold blue",
    )

    table.add_column("Group", style="dim", width=8)
    table.add_column("Type", style="bold", width=10)
    table.add_column("Parameters", style="cyan", width=12)
    table.add_column("Learning Rate", style="yellow", width=15)
    table.add_column("Config", style="green", width=30)

    for i, group in enumerate(optimizer.param_groups):
        if hasattr(group, "__contains__") and "use_muon" in group:
            opt_type = "Muon" if group["use_muon"] else "AdamW"
            num_params = len(group["params"])
            lr = group["lr"]

            if group["use_muon"]:
                config = (
                    f"momentum={group['momentum']}, ns_steps={group.get('ns_steps', 5)}"
                )
            else:
                config = (
                    f"betas={group.get('betas', 'N/A')}, eps={group.get('eps', 'N/A')}"
                )
        else:
            # SingleDeviceMuon case
            opt_type = "Muon"
            num_params = len(group["params"])
            lr = group["lr"]
            config = (
                f"momentum={group['momentum']}, ns_steps={group.get('ns_steps', 5)}"
            )

        table.add_row(
            str(i + 1),
            opt_type,
            str(num_params),
            f"{lr:.2e}" if lr < 0.001 else f"{lr:.4f}",
            config,
        )

    rich_console().print(table)


def example_1_basic_muon():
    """Example 1: Basic usage with SingleDeviceMuon."""
    log().info("\n[bold cyan]📚 Example 1: Basic Muon Optimizer[/bold cyan]")

    # Create model and data
    model = SimpleNet()
    X, y = create_dummy_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize Muon optimizer
    optimizer = SingleDeviceMuon(
        model.parameters(), lr=0.02, momentum=0.95, weight_decay=0.01
    )

    log().info(
        f"[yellow]Model[/yellow]: {type(model).__name__} with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Show model structure
    create_model_summary_table(model, "SingleDeviceMuon")
    create_optimizer_config_table(optimizer)

    # Train the model
    model, metrics = train_model(model, optimizer, dataloader, num_epochs=3)

    # Show training results
    create_training_results_table(metrics, "Training Results - Basic Muon")

    log().info("[green]✅ Example 1 completed![/green]")
    return model, optimizer, metrics


def example_2_hybrid_optimization():
    """Example 2: Hybrid optimization with Muon and AdamW."""
    log().info(
        "\n[bold cyan]📚 Example 2: Hybrid Muon + AdamW Optimization[/bold cyan]"
    )

    # Create model and data
    model = SimpleNet()
    X, y = create_dummy_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create parameter groups automatically
    param_groups = create_muon_param_groups(
        model, muon_lr=0.02, adam_lr=3e-4, muon_momentum=0.95, weight_decay=0.01
    )

    # Initialize hybrid optimizer
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    log().info(
        f"[yellow]Model[/yellow]: {type(model).__name__} with hybrid optimization"
    )

    # Show configuration
    create_model_summary_table(model, "Hybrid Muon+AdamW")
    create_optimizer_config_table(optimizer)

    # Train the model
    model, metrics = train_model(model, optimizer, dataloader, num_epochs=3)

    # Show training results
    create_training_results_table(metrics, "Training Results - Hybrid Optimization")

    log().info("[green]✅ Example 2 completed![/green]")
    return model, optimizer, metrics


def example_3_manual_parameter_grouping():
    """Example 3: Manual parameter grouping."""
    log().info("\n[bold cyan]📚 Example 3: Manual Parameter Grouping[/bold cyan]")

    # Create model and data
    model = SimpleNet()
    X, y = create_dummy_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Manually separate parameters
    matrix_params = []
    scalar_params = []

    param_analysis = []
    for name, param in model.named_parameters():
        param_info = {
            "name": name,
            "shape": tuple(param.shape),
            "elements": param.numel(),
            "dimensions": param.ndim,
            "type": "Matrix" if param.ndim >= 2 else "Vector",
            "optimizer": "Muon" if param.ndim >= 2 else "AdamW",
        }
        param_analysis.append(param_info)

        if param.ndim >= 2:
            matrix_params.append(param)
        else:
            scalar_params.append(param)

    # Create parameter analysis table
    table = Table(
        title="Manual Parameter Analysis",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Parameter", style="bold", width=18)
    table.add_column("Shape", style="cyan", width=16)
    table.add_column("Elements", style="yellow", width=12)
    table.add_column("Dims", style="blue", width=8)
    table.add_column("Assignment", style="green", width=12)

    for info in param_analysis:
        assignment_color = "green" if info["optimizer"] == "Muon" else "yellow"
        table.add_row(
            info["name"],
            str(info["shape"]),
            f"{info['elements']:,}",
            str(info["dimensions"]),
            f"[{assignment_color}]{info['optimizer']}[/{assignment_color}]",
        )

    rich_console().print(table)

    # Create parameter groups manually
    param_groups = [
        # Muon for matrix parameters
        {
            "params": matrix_params,
            "lr": 0.02,
            "momentum": 0.95,
            "weight_decay": 0.01,
            "ns_steps": 5,
            "use_muon": True,
        },
        # AdamW for scalar parameters
        {
            "params": scalar_params,
            "lr": 3e-4,
            "betas": (0.9, 0.95),
            "eps": 1e-10,
            "weight_decay": 0.01,
            "use_muon": False,
        },
    ]

    # Initialize hybrid optimizer
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    create_optimizer_config_table(optimizer)

    # Train the model
    model, metrics = train_model(model, optimizer, dataloader, num_epochs=3)

    # Show training results
    create_training_results_table(metrics, "Training Results - Manual Grouping")

    log().info("[green]✅ Example 3 completed![/green]")
    return model, optimizer, metrics


def example_4_convolutional_network():
    """Example 4: Training a convolutional network with Muon."""
    log().info("\n[bold cyan]📚 Example 4: Convolutional Network with Muon[/bold cyan]")

    # Create model and data
    model = ConvNet()
    X, y = create_dummy_data(num_samples=500, input_size=28 * 28)
    X = X.view(-1, 1, 28, 28)  # Reshape for conv layers
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Muon automatically handles conv layers by reshaping to 2D
    optimizer = SingleDeviceMuon(
        model.parameters(), lr=0.02, momentum=0.95, weight_decay=0.01
    )

    log().info(
        f"[yellow]Model[/yellow]: ConvNet with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Show how conv layers are handled
    conv_analysis = []
    for name, param in model.named_parameters():
        layer_type = (
            "Convolutional" if "conv" in name else "Linear" if "fc" in name else "Other"
        )
        muon_compatible = "Yes" if param.ndim >= 2 else "No"

        conv_analysis.append(
            {
                "name": name,
                "shape": tuple(param.shape),
                "type": layer_type,
                "compatible": muon_compatible,
                "elements": param.numel(),
            }
        )

    table = Table(
        title="ConvNet Layer Analysis",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Layer", style="bold", width=18)
    table.add_column("Shape", style="cyan", width=16)
    table.add_column("Type", style="yellow", width=14)
    table.add_column("Muon Compatible", style="green", width=16)
    table.add_column("Elements", style="blue", width=12)

    for info in conv_analysis:
        compat_color = "green" if info["compatible"] == "Yes" else "red"
        table.add_row(
            info["name"],
            str(info["shape"]),
            info["type"],
            f"[{compat_color}]{info['compatible']}[/{compat_color}]",
            f"{info['elements']:,}",
        )

    rich_console().print(table)
    create_optimizer_config_table(optimizer)

    # Train the model
    model, metrics = train_model(model, optimizer, dataloader, num_epochs=3)

    # Show training results
    create_training_results_table(metrics, "Training Results - ConvNet")

    log().info("[green]✅ Example 4 completed![/green]")
    return model, optimizer, metrics


def example_5_learning_rate_scheduling():
    """Example 5: Using Muon with learning rate scheduling."""
    log().info(
        "\n[bold cyan]📚 Example 5: Muon with Learning Rate Scheduling[/bold cyan]"
    )

    # Create model and data
    model = SimpleNet()
    X, y = create_dummy_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize optimizer and scheduler
    optimizer = SingleDeviceMuon(
        model.parameters(), lr=0.02, momentum=0.95, weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    log().info(
        f"[yellow]Initial learning rate[/yellow]: {optimizer.param_groups[0]['lr']:.6f}"
    )
    log().info(f"[yellow]Scheduler[/yellow]: StepLR (step_size=1, gamma=0.9)")

    # Train with scheduler and track LR changes
    model.train()
    criterion = nn.CrossEntropyLoss()

    lr_schedule = []
    training_metrics = []

    for epoch in range(3):
        epoch_start = time.time()
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        current_lr = optimizer.param_groups[0]["lr"]
        lr_schedule.append(current_lr)

        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        # Step the scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        accuracy = 100.0 * correct / total

        training_metrics.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "accuracy": accuracy,
                "lr": current_lr,
                "time": epoch_time,
            }
        )

    # Create LR schedule table
    table = Table(
        title="Learning Rate Schedule",
        show_header=True,
        header_style="bold yellow",
    )

    table.add_column("Epoch", style="dim", width=8)
    table.add_column("Learning Rate", style="yellow", width=15)
    table.add_column("Loss", style="red", width=12)
    table.add_column("Accuracy", style="green", width=12)
    table.add_column("Time (s)", style="cyan", width=10)

    for metric in training_metrics:
        table.add_row(
            str(metric["epoch"]),
            f"{metric['lr']:.6f}",
            f"{metric['loss']:.4f}",
            f"{metric['accuracy']:.2f}%",
            f"{metric['time']:.2f}",
        )

    rich_console().print(table)

    log().info("[green]✅ Example 5 completed![/green]")
    return model, optimizer, training_metrics


def example_6_parameter_analysis():
    """Example 6: Analyze parameter types and demonstrate grouping logic."""
    log().info("\n[bold cyan]📚 Example 6: Parameter Analysis Comparison[/bold cyan]")

    models = {"SimpleNet (MLP)": SimpleNet(), "ConvNet (CNN)": ConvNet()}

    # Create comprehensive comparison table
    comparison_table = Table(
        title="Model Architecture Comparison",
        show_header=True,
        header_style="bold magenta",
    )

    comparison_table.add_column("Model", style="bold", width=16)
    comparison_table.add_column("Total Params", style="cyan", width=14)
    comparison_table.add_column("Muon Params", style="green", width=14)
    comparison_table.add_column("AdamW Params", style="yellow", width=14)
    comparison_table.add_column("Muon %", style="blue", width=10)
    comparison_table.add_column("Layers", style="magenta", width=12)

    detailed_analysis = {}

    for model_name, model in models.items():
        total_params = 0
        muon_params = 0
        adam_params = 0
        layer_count = 0

        layer_details = []

        for name, param in model.named_parameters():
            total_params += param.numel()
            layer_count += 1

            if param.ndim >= 2:
                muon_params += param.numel()
                opt_type = "Muon"
            else:
                adam_params += param.numel()
                opt_type = "AdamW"

            layer_details.append(
                {
                    "name": name,
                    "shape": tuple(param.shape),
                    "elements": param.numel(),
                    "optimizer": opt_type,
                }
            )

        muon_percentage = 100.0 * muon_params / total_params if total_params > 0 else 0

        comparison_table.add_row(
            model_name,
            f"{total_params:,}",
            f"{muon_params:,}",
            f"{adam_params:,}",
            f"{muon_percentage:.1f}%",
            str(layer_count),
        )

        detailed_analysis[model_name] = layer_details

    rich_console().print(comparison_table)

    # Show detailed breakdown for each model
    for model_name, details in detailed_analysis.items():
        table = Table(
            title=f"Detailed Layer Breakdown - {model_name}",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Layer", style="bold", width=20)
        table.add_column("Shape", style="cyan", width=16)
        table.add_column("Elements", style="yellow", width=12)
        table.add_column("Optimizer", style="green", width=12)

        for detail in details:
            opt_color = "green" if detail["optimizer"] == "Muon" else "yellow"
            table.add_row(
                detail["name"],
                str(detail["shape"]),
                f"{detail['elements']:,}",
                f"[{opt_color}]{detail['optimizer']}[/{opt_color}]",
            )

        rich_console().print(table)

    log().info("[green]✅ Example 6 completed![/green]")
    return models


def run_all_examples():
    """Run all examples sequentially."""
    log().info("[bold cyan]🚀 Muon Optimizer Examples[/bold cyan]")
    log().info(
        "[yellow]Comprehensive demonstration of Muon optimizer usage patterns[/yellow]"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log().info(f"[yellow]Using device[/yellow]: {device}")
    if device == "cpu":
        log().info("[dim]Note: Running on CPU. For better performance, use GPU.[/dim]")

    results = {}

    try:
        # Run examples
        results["basic"] = example_1_basic_muon()
        results["hybrid"] = example_2_hybrid_optimization()
        results["manual"] = example_3_manual_parameter_grouping()
        results["conv"] = example_4_convolutional_network()
        results["scheduler"] = example_5_learning_rate_scheduling()
        results["analysis"] = example_6_parameter_analysis()

        # Create final summary table
        log().info("\n[bold magenta]📊 Final Summary[/bold magenta]")
        create_final_summary_table(results)

        log().info("[bold green]✅ All examples completed successfully![/bold green]")

    except Exception as e:
        log().error(f"[red]❌ Error running examples: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise

    return results


def create_final_summary_table(results: Dict) -> None:
    """Create a final summary table of all examples."""
    table = Table(
        title="Example Execution Summary",
        show_header=True,
        header_style="bold green",
    )

    table.add_column("Example", style="bold", width=20)
    table.add_column("Model", style="cyan", width=16)
    table.add_column("Optimizer", style="yellow", width=20)
    table.add_column("Parameters", style="blue", width=12)
    table.add_column("Final Accuracy", style="green", width=14)
    table.add_column("Status", style="magenta", width=10)

    for name, result in results.items():
        if name == "analysis":
            # Special case for analysis example
            table.add_row(
                "Parameter Analysis",
                "Multiple",
                "Comparison",
                "Various",
                "N/A",
                "✅ Complete",
            )
        elif isinstance(result, tuple) and len(result) >= 3:
            model, optimizer, metrics = result[:3]
            param_count = sum(p.numel() for p in model.parameters())
            final_accuracy = metrics[-1]["accuracy"] if metrics else 0.0

            table.add_row(
                name.replace("_", " ").title(),
                type(model).__name__,
                type(optimizer).__name__.replace("SingleDevice", ""),
                f"{param_count:,}",
                f"{final_accuracy:.2f}%",
                "✅ Complete",
            )

    rich_console().print(table)


def main():
    """Main function to demonstrate example usage."""
    run_all_examples()

    log().info("\n[bold cyan]🎉 Example usage demonstration completed![/bold cyan]")
    log().info(
        "[dim]Check the rich tables above for detailed results and analysis.[/dim]"
    )


@functools.cache
def log():
    """Get logger instance."""
    return logging.getLogger("example_usage")


@functools.cache
def rich_console():
    """Get rich console instance."""
    return Console()


if __name__ == "__main__":
    # Configure rich logging
    rich.traceback.install(show_locals=False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=rich_console(), rich_tracebacks=True, markup=True)
        ],
    )

    main()
