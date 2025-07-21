"""
MNIST Optimizer Benchmark: Muon vs SGD vs Adam
Comprehensive comparison using neural network training on MNIST dataset.
"""

import functools
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple

from muon_optimizer import Muon
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
import rich.traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTNet(nn.Module):
    """Simple CNN for MNIST classification."""

    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class OptimizationTracker:
    """Track optimization metrics during training."""

    def __init__(self, name: str):
        self.name = name
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.epoch_times = []
        self.best_test_acc = 0.0
        self.convergence_epoch = None

    def update(
        self,
        train_loss: float,
        train_acc: float,
        test_acc: float,
        epoch_time: float,
        epoch: int,
    ):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)
        self.epoch_times.append(epoch_time)

        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc

        # Check for convergence (95% of best accuracy reached)
        if self.convergence_epoch is None and test_acc >= 0.95 * max(
            self.test_accuracies
        ):
            self.convergence_epoch = epoch

    def get_final_metrics(self) -> Dict:
        return {
            "final_test_acc": self.test_accuracies[-1] if self.test_accuracies else 0.0,
            "best_test_acc": self.best_test_acc,
            "final_train_loss": (
                self.train_losses[-1] if self.train_losses else float("inf")
            ),
            "convergence_epoch": self.convergence_epoch or len(self.test_accuracies),
            "avg_epoch_time": np.mean(self.epoch_times) if self.epoch_times else 0.0,
            "total_time": sum(self.epoch_times) if self.epoch_times else 0.0,
        }


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer,
    epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Simple training loop without nested progress bars
    running_loss = 0.0
    total_batches = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Log progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_running_loss = running_loss / 100
            log().info(
                f"[dim]Batch {batch_idx + 1}/{total_batches} - Loss: {avg_running_loss:.4f}[/dim]"
            )
            running_loss = 0.0

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def test_model(
    model: nn.Module, device: torch.device, test_loader: DataLoader
) -> float:
    """Test model and return accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def load_mnist_data():
    """Load MNIST data"""
    try:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Use XDG_DATA_HOME or default to ~/.local/share
        xdg_data_home = os.environ.get(
            "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
        )
        mnist_data_dir = os.path.join(xdg_data_home, "mnist_benchmark_data")

        train_dataset = datasets.MNIST(
            mnist_data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            mnist_data_dir,
            download=True,
            train=False,
            transform=transform,
        )

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        return train_loader, test_loader, "Real MNIST"
    except Exception as e:
        log().warning(f"Failed to load real MNIST: {e}")


def train_with_optimizer(
    optimizer_name: str,
    optimizer_class,
    optimizer_kwargs: Dict,
    epochs: int = 10,
    device: torch.device = None,
) -> OptimizationTracker:
    """Train a model with specified optimizer and track metrics."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader, data_type = load_mnist_data()

    # Model and optimizer setup
    model = MNISTNet().to(device)

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    tracker = OptimizationTracker(optimizer_name)

    log().info(f"\n[bold cyan]🚀 Training with {optimizer_name}[/bold cyan]")
    log().info(f"[yellow]Device[/yellow]: {device}")
    log().info(f"[yellow]Data type[/yellow]: {data_type}")
    log().info(
        f"[yellow]Model parameters[/yellow]: {sum(p.numel() for p in model.parameters()):,}"
    )
    log().info(f"[yellow]Optimizer config[/yellow]: {optimizer_kwargs}")

    # Training loop
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=rich_console(),
    ) as progress:
        task = progress.add_task(f"Training {optimizer_name}", total=epochs)

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = train_epoch(
                model, device, train_loader, optimizer, epoch
            )

            # Test
            test_acc = test_model(model, device, test_loader)

            epoch_time = time.time() - epoch_start
            tracker.update(train_loss, train_acc, test_acc, epoch_time, epoch + 1)

            # Log progress
            if epoch % 2 == 0 or epoch == epochs - 1:
                log().info(
                    f"[dim]Epoch {epoch + 1:2d}[/dim] | "
                    f"[red]Loss[/red]: {train_loss:.4f} | "
                    f"[green]Train Acc[/green]: {train_acc:.2f}% | "
                    f"[blue]Test Acc[/blue]: {test_acc:.2f}% | "
                    f"[yellow]Time[/yellow]: {epoch_time:.2f}s"
                )

            progress.advance(task)

    # Final summary
    metrics = tracker.get_final_metrics()
    log().info(f"[bold green]✅ {optimizer_name} Training Complete[/bold green]")
    log().info(f"[cyan]Best Test Accuracy[/cyan]: {metrics['best_test_acc']:.2f}%")
    log().info(f"[cyan]Total Training Time[/cyan]: {metrics['total_time']:.1f}s")

    return tracker


def create_results_table(trackers: List[OptimizationTracker]) -> None:
    """Create comprehensive results comparison table."""

    table = Table(
        title="MNIST Optimizer Benchmark Results",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Optimizer", style="bold", width=12)
    table.add_column("Final Test Acc", style="green", width=14)
    table.add_column("Best Test Acc", style="cyan", width=13)
    table.add_column("Final Train Loss", style="yellow", width=15)
    table.add_column("Convergence Epoch", style="blue", width=16)
    table.add_column("Avg Epoch Time", style="red", width=14)
    table.add_column("Total Time", style="magenta", width=12)
    table.add_column("Ranking", style="bright_green", width=10)

    # Sort by best test accuracy
    sorted_trackers = sorted(
        trackers, key=lambda t: t.get_final_metrics()["best_test_acc"], reverse=True
    )

    for rank, tracker in enumerate(sorted_trackers, 1):
        metrics = tracker.get_final_metrics()
        rank_emoji = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"{rank}th"

        table.add_row(
            tracker.name,
            f"{metrics['final_test_acc']:.2f}%",
            f"{metrics['best_test_acc']:.2f}%",
            f"{metrics['final_train_loss']:.4f}",
            f"{metrics['convergence_epoch']}",
            f"{metrics['avg_epoch_time']:.2f}s",
            f"{metrics['total_time']:.1f}s",
            rank_emoji,
        )

    rich_console().print(table)

    # Winner announcement
    winner = sorted_trackers[0]
    winner_metrics = winner.get_final_metrics()
    log().info(f"\n[bold green]🏆 Winner: {winner.name}[/bold green]")
    log().info(
        f"[green]Best Test Accuracy[/green]: {winner_metrics['best_test_acc']:.2f}%"
    )
    log().info(
        f"[green]Convergence Speed[/green]: {winner_metrics['convergence_epoch']} epochs"
    )


def create_detailed_progress_table(trackers: List[OptimizationTracker]) -> None:
    """Create detailed epoch-by-epoch progress comparison."""

    table = Table(
        title="Epoch-by-Epoch Test Accuracy Progress",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Epoch", style="bold", width=8)
    for tracker in trackers:
        table.add_column(tracker.name, style="cyan", width=10)

    max_epochs = max(len(tracker.test_accuracies) for tracker in trackers)

    # Show every 2nd epoch for readability
    for epoch in range(0, max_epochs, 2):
        row = [str(epoch + 1)]
        for tracker in trackers:
            if epoch < len(tracker.test_accuracies):
                row.append(f"{tracker.test_accuracies[epoch]:.2f}%")
            else:
                row.append("-")
        table.add_row(*row)

    # Add final epoch if not already shown
    if (max_epochs - 1) % 2 != 0:
        row = [str(max_epochs)]
        for tracker in trackers:
            if max_epochs <= len(tracker.test_accuracies):
                row.append(f"{tracker.test_accuracies[max_epochs - 1]:.2f}%")
            else:
                row.append("-")
        table.add_row(*row)

    rich_console().print(table)


def main():
    """Run comprehensive MNIST optimizer benchmark."""

    log().info("[bold cyan]🎯 MNIST Neural Network Optimizer Benchmark[/bold cyan]")
    log().info("[yellow]Comparing Muon vs SGD vs Adam on real MNIST dataset[/yellow]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log().info(f"[yellow]Using device[/yellow]: {device}")

    epochs = 15

    # Optimizer configurations (tuned for fair comparison)
    optimizer_configs = [
        {
            "name": "SGD",
            "class": torch.optim.SGD,
            "kwargs": {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4},
        },
        {
            "name": "Adam",
            "class": torch.optim.Adam,
            "kwargs": {"lr": 0.001, "weight_decay": 1e-4},
        },
        {
            "name": "Muon",
            "class": Muon,
            "kwargs": {
                "lr": 0.005,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "ns_steps": 3,
            },
        },
    ]

    log().info(f"[yellow]Training epochs[/yellow]: {epochs}")
    log().info(f"[yellow]Batch size[/yellow]: 128")

    # Train with each optimizer
    trackers = []
    for config in optimizer_configs:
        tracker = train_with_optimizer(
            config["name"],
            config["class"],
            config["kwargs"],
            epochs=epochs,
            device=device,
        )
        trackers.append(tracker)

    # Create comparison tables
    log().info("\n[bold magenta]📊 Benchmark Results Analysis[/bold magenta]")
    create_results_table(trackers)

    log().info("\n[bold magenta]📈 Detailed Progress Analysis[/bold magenta]")
    create_detailed_progress_table(trackers)

    # Performance insights
    log().info("\n[bold magenta]💡 Performance Insights[/bold magenta]")

    best_acc_tracker = max(
        trackers, key=lambda t: t.get_final_metrics()["best_test_acc"]
    )
    fastest_tracker = min(
        trackers, key=lambda t: t.get_final_metrics()["convergence_epoch"]
    )
    fastest_time_tracker = min(
        trackers, key=lambda t: t.get_final_metrics()["avg_epoch_time"]
    )

    log().info(
        f"[green]🎯 Highest Accuracy[/green]: {best_acc_tracker.name} ({best_acc_tracker.get_final_metrics()['best_test_acc']:.2f}%)"
    )
    log().info(
        f"[blue]⚡ Fastest Convergence[/blue]: {fastest_tracker.name} ({fastest_tracker.get_final_metrics()['convergence_epoch']} epochs)"
    )
    log().info(
        f"[yellow]⏱️  Fastest Per Epoch[/yellow]: {fastest_time_tracker.name} ({fastest_time_tracker.get_final_metrics()['avg_epoch_time']:.2f}s/epoch)"
    )

    log().info("[dim]MNIST benchmark completed[/dim]")


@functools.cache
def log():
    return logging.getLogger("mnist_benchmark")


@functools.cache
def rich_console():
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
