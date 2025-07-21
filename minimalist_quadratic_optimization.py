"""
Minimalist 2D Quadratic Optimization
"""

import functools
import logging
import time
from typing import Tuple

import numpy as np
import rich.traceback
import torch
import torch.nn as nn
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from muon_optimizer import Muon


class OptimizationProblem(nn.Module):
    """Base class for 2D optimization problems."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.x_min = None
        self.y_min = None
        self.f_min = None

    def forward(self, params):
        x, y = params[0], params[1]
        return self.compute_loss(x, y)

    def compute_loss(self, x, y):
        raise NotImplementedError


class QuadraticLoss(OptimizationProblem):
    """Simple 2D quadratic loss function: f(x, y) = a*x² + b*y² + c*xy + d*x + e*y + f"""

    def __init__(self, a=1.0, b=4.0, c=0.5, d=-2.0, e=-8.0, f=5.0):
        super().__init__("Quadratic")
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f

        # Analytical minimum
        det = 4 * a * b - c * c
        if abs(det) > 1e-10:
            self.x_min = (2 * b * d - c * e) / det
            self.y_min = (2 * a * e - c * d) / det
        else:
            self.x_min = self.y_min = 0.0

        self.f_min = self.compute_loss(self.x_min, self.y_min)
        log().info(
            f"[cyan]{self.name} minimum[/cyan]: ({self.x_min:.4f}, {self.y_min:.4f}), loss: {self.f_min:.4f}"
        )

    def compute_loss(self, x, y):
        return (
            self.a * x**2
            + self.b * y**2
            + self.c * x * y
            + self.d * x
            + self.e * y
            + self.f
        )


class RosenbrockFunction(OptimizationProblem):
    """Rosenbrock function: f(x,y) = a*(y-x²)² + (1-x)²"""

    def __init__(self, a=100.0):
        super().__init__("Rosenbrock")
        self.a = a
        self.x_min = 1.0
        self.y_min = 1.0
        self.f_min = 0.0
        log().info(
            f"[cyan]{self.name} minimum[/cyan]: ({self.x_min:.4f}, {self.y_min:.4f}), loss: {self.f_min:.4f}"
        )

    def compute_loss(self, x, y):
        return self.a * (y - x**2) ** 2 + (1 - x) ** 2


class BealeFunction(OptimizationProblem):
    """Beale function: f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)²"""

    def __init__(self):
        super().__init__("Beale")
        self.x_min = 3.0
        self.y_min = 0.5
        self.f_min = 0.0
        log().info(
            f"[cyan]{self.name} minimum[/cyan]: ({self.x_min:.4f}, {self.y_min:.4f}), loss: {self.f_min:.4f}"
        )

    def compute_loss(self, x, y):
        term1 = (1.5 - x + x * y) ** 2
        term2 = (2.25 - x + x * y**2) ** 2
        term3 = (2.625 - x + x * y**3) ** 2
        return term1 + term2 + term3


class IllConditionedQuadratic(OptimizationProblem):
    """Ill-conditioned quadratic with high condition number"""

    def __init__(self, condition_number=1000.0):
        super().__init__(f"Ill-Conditioned (κ={condition_number:.0f})")
        self.a = condition_number
        self.b = 1.0
        self.x_min = 0.0
        self.y_min = 0.0
        self.f_min = 0.0
        log().info(
            f"[cyan]{self.name} minimum[/cyan]: ({self.x_min:.4f}, {self.y_min:.4f}), loss: {self.f_min:.4f}"
        )

    def compute_loss(self, x, y):
        return 0.5 * (self.a * x**2 + self.b * y**2)


class SaddlePoint(OptimizationProblem):
    """Saddle point function: f(x,y) = x² - y²"""

    def __init__(self):
        super().__init__("Saddle Point")
        self.x_min = 0.0
        self.y_min = 0.0
        self.f_min = 0.0
        log().info(
            f"[cyan]{self.name} critical point[/cyan]: ({self.x_min:.4f}, {self.y_min:.4f}), loss: {self.f_min:.4f}"
        )

    def compute_loss(self, x, y):
        return x**2 - y**2


class HimmelblauFunction(OptimizationProblem):
    """Himmelblau's function: f(x,y) = (x²+y-11)² + (x+y²-7)² (multiple minima)"""

    def __init__(self):
        super().__init__("Himmelblau")
        # One of four global minima
        self.x_min = 3.0
        self.y_min = 2.0
        self.f_min = 0.0
        log().info(
            f"[cyan]{self.name} minimum (1 of 4)[/cyan]: ({self.x_min:.4f}, {self.y_min:.4f}), loss: {self.f_min:.4f}"
        )

    def compute_loss(self, x, y):
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def optimize_and_log(
    optimizer_name: str,
    params: torch.Tensor,
    optimizer,
    loss_fn: OptimizationProblem,
    steps: int = 20,
):
    """Run optimization and log trajectory points."""

    # Get initial position
    if params.dim() == 1:
        get_pos = lambda: (params[0].item(), params[1].item())
        loss_func = lambda: loss_fn(params)
    else:  # Muon case (2D tensor)
        get_pos = lambda: (params[0, 0].item(), params[0, 1].item())
        loss_func = lambda: loss_fn(params.view(-1))

    # Log initial state
    x, y = get_pos()
    initial_loss = loss_func().item()
    distance_to_min = np.sqrt((x - loss_fn.x_min) ** 2 + (y - loss_fn.y_min) ** 2)

    log().info(
        f"\n[bold {['red', 'blue', 'green'][hash(optimizer_name) % 3]}]{optimizer_name} Optimization[/bold {['red', 'blue', 'green'][hash(optimizer_name) % 3]}]"
    )
    log().info(
        f"[yellow]Initial[/yellow]: ({x:.4f}, {y:.4f}) | Loss: {initial_loss:.4f} | Distance: {distance_to_min:.4f}"
    )

    # Create trajectory table
    table = Table(
        title=f"{optimizer_name} Trajectory",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Step", style="dim", width=6)
    table.add_column("X", style="cyan", width=10)
    table.add_column("Y", style="cyan", width=10)
    table.add_column("Loss", style="yellow", width=12)
    table.add_column("Step Size", style="green", width=10)
    table.add_column("Distance", style="red", width=10)

    # Add initial row
    table.add_row(
        "0",
        f"{x:.4f}",
        f"{y:.4f}",
        f"{initial_loss:.4f}",
        "0.0000",
        f"{distance_to_min:.4f}",
    )

    # Run optimization steps
    start_time = time.time()
    prev_x, prev_y = x, y

    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_func()
        loss.backward()
        optimizer.step()

        # Get new position
        x, y = get_pos()
        step_size = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
        distance_to_min = np.sqrt((x - loss_fn.x_min) ** 2 + (y - loss_fn.y_min) ** 2)

        # Add to table (show every 2nd step to keep output manageable)
        if step % 2 == 0 or step < 5 or step >= steps - 3:
            table.add_row(
                str(step + 1),
                f"{x:.4f}",
                f"{y:.4f}",
                f"{loss.item():.4f}",
                f"{step_size:.4f}",
                f"{distance_to_min:.4f}",
            )

        prev_x, prev_y = x, y

    elapsed_time = time.time() - start_time

    # Display trajectory table
    rich_console().print(table)

    # Final summary
    final_loss = loss.item()
    total_reduction = initial_loss - final_loss
    log().info(
        f"[bold green]Final[/bold green]: ({x:.4f}, {y:.4f}) | Loss: {final_loss:.4f} | Reduction: {total_reduction:.4f}"
    )
    log().info(f"[dim]Completed {steps} steps in {elapsed_time:.3f}s[/dim]")

    return x, y, final_loss


def run_optimizer_on_problem(
    problem: OptimizationProblem,
    initial_point: Tuple[float, float],
    learning_rates: dict,
    steps: int = 25,
):
    """Run all optimizers on a single problem and return results."""
    results = {}

    log().info(f"\n[bold cyan]🎯 Testing {problem.name} Function[/bold cyan]")

    # SGD Optimization
    params_sgd = torch.tensor(initial_point, requires_grad=True, dtype=torch.float32)
    optimizer_sgd = torch.optim.SGD([params_sgd], lr=learning_rates["SGD"])
    results["SGD"] = optimize_and_log("SGD", params_sgd, optimizer_sgd, problem, steps)

    # Adam Optimization
    params_adam = torch.tensor(initial_point, requires_grad=True, dtype=torch.float32)
    optimizer_adam = torch.optim.Adam([params_adam], lr=learning_rates["Adam"])
    results["Adam"] = optimize_and_log(
        "Adam", params_adam, optimizer_adam, problem, steps
    )

    # Muon Optimization
    params_muon = torch.tensor([initial_point], requires_grad=True, dtype=torch.float32)
    optimizer_muon = Muon([params_muon], lr=learning_rates["Muon"], momentum=0.9)
    results["Muon"] = optimize_and_log(
        "Muon", params_muon, optimizer_muon, problem, steps
    )

    return results


def create_comparison_table(results: dict, problem: OptimizationProblem):
    """Create a comparison table for optimizer results on a single problem."""
    comparison_table = Table(
        title=f"{problem.name} - Optimizer Comparison",
        show_header=True,
        header_style="bold cyan",
    )
    comparison_table.add_column("Optimizer", style="bold", width=12)
    comparison_table.add_column("Final X", style="cyan", width=10)
    comparison_table.add_column("Final Y", style="cyan", width=10)
    comparison_table.add_column("Final Loss", style="yellow", width=12)
    comparison_table.add_column("Distance to Min", style="red", width=15)
    comparison_table.add_column("Ranking", style="green", width=10)

    # Sort by final loss
    sorted_results = sorted(results.items(), key=lambda x: x[1][2])

    for rank, (name, (x, y, loss)) in enumerate(sorted_results, 1):
        if problem.x_min is not None and problem.y_min is not None:
            distance = np.sqrt((x - problem.x_min) ** 2 + (y - problem.y_min) ** 2)
        else:
            distance = 0.0
        rank_emoji = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"{rank}th"

        comparison_table.add_row(
            name, f"{x:.4f}", f"{y:.4f}", f"{loss:.4f}", f"{distance:.4f}", rank_emoji
        )

    rich_console().print(comparison_table)

    # Winner announcement
    winner = sorted_results[0][0]
    log().info(f"[bold green]🏆 {problem.name} Winner: {winner}[/bold green]")

    return sorted_results


def main():
    """Run comprehensive 2D optimization benchmark across multiple problems."""
    log().info("[bold cyan]🎯 Comprehensive 2D Optimization Benchmark[/bold cyan]")

    # Define optimization problems with different characteristics
    problems = [
        QuadraticLoss(a=1.0, b=4.0, c=0.5, d=-2.0, e=-8.0, f=5.0),  # Simple quadratic
        IllConditionedQuadratic(condition_number=1000.0),  # Ill-conditioned
        RosenbrockFunction(a=100.0),  # Non-convex, narrow valley
        BealeFunction(),  # Non-convex, complex landscape
        SaddlePoint(),  # Saddle point
        HimmelblauFunction(),  # Multiple minima
    ]

    # Problem-specific starting points and learning rates
    problem_configs = {
        "Quadratic": {
            "start": (3.0, 3.0),
            "lr": {"SGD": 0.02, "Adam": 0.1, "Muon": 0.03},
        },
        "Ill-Conditioned (κ=1000)": {
            "start": (10.0, 10.0),
            "lr": {"SGD": 0.001, "Adam": 0.01, "Muon": 0.005},
        },
        "Rosenbrock": {
            "start": (-1.0, 1.0),
            "lr": {"SGD": 0.001, "Adam": 0.01, "Muon": 0.005},
        },
        "Beale": {
            "start": (1.0, 1.0),
            "lr": {"SGD": 0.001, "Adam": 0.01, "Muon": 0.005},
        },
        "Saddle Point": {
            "start": (1.0, 1.0),
            "lr": {"SGD": 0.01, "Adam": 0.05, "Muon": 0.02},
        },
        "Himmelblau": {
            "start": (0.0, 0.0),
            "lr": {"SGD": 0.005, "Adam": 0.02, "Muon": 0.01},
        },
    }

    steps = 30
    all_results = {}

    # Run benchmark on each problem
    for problem in problems:
        config = problem_configs[problem.name]
        initial_point = config["start"]
        learning_rates = config["lr"]

        log().info(f"[yellow]Starting point[/yellow]: {initial_point}")
        log().info(f"[yellow]Learning rates[/yellow]: {learning_rates}")
        log().info(f"[yellow]Steps per optimizer[/yellow]: {steps}")

        results = run_optimizer_on_problem(
            problem, initial_point, learning_rates, steps
        )
        all_results[problem.name] = create_comparison_table(results, problem)

    # Create overall summary
    log().info("\n[bold magenta]📊 Overall Performance Summary[/bold magenta]")

    summary_table = Table(
        title="Optimizer Performance Across All Problems",
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Problem", style="bold", width=20)
    summary_table.add_column("Winner", style="green", width=12)
    summary_table.add_column("2nd Place", style="yellow", width=12)
    summary_table.add_column("3rd Place", style="red", width=12)

    optimizer_wins = {"SGD": 0, "Adam": 0, "Muon": 0}

    for problem_name, sorted_results in all_results.items():
        winner = sorted_results[0][0]
        second = sorted_results[1][0]
        third = sorted_results[2][0]

        optimizer_wins[winner] += 1

        summary_table.add_row(
            problem_name, f"🥇 {winner}", f"🥈 {second}", f"🥉 {third}"
        )

    rich_console().print(summary_table)

    # Overall champion
    champion = max(optimizer_wins.items(), key=lambda x: x[1])
    log().info(
        f"\n[bold green]🏆 Overall Champion: {champion[0]}[/bold green] ({champion[1]} wins)"
    )
    log().info(f"[yellow]Win distribution[/yellow]: {optimizer_wins}")
    log().info("[dim]Comprehensive benchmark completed[/dim]")


@functools.cache
def log():
    return logging.getLogger("minimalist_optimization")


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
