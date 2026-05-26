#!/usr/bin/env python3
"""Export static JSON bundles for the Muon Optimizer Explorer UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minimalist_quadratic_optimization import (  # noqa: E402
    BealeFunction,
    HimmelblauFunction,
    IllConditionedQuadratic,
    OptimizationProblem,
    QuadraticLoss,
    RosenbrockFunction,
    SaddlePoint,
)
from mnist_optimizer_benchmark import (  # noqa: E402
    OptimizationTracker,
    train_with_optimizer,
)
from muon_optimizer import Muon  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "explorer" / "public" / "data"
GRID_SIZE = 50

LITE_PROBLEMS = ["quadratic", "rosenbrock", "ill_conditioned"]
FULL_PROBLEMS = [
    "quadratic",
    "ill_conditioned",
    "rosenbrock",
    "beale",
    "saddle_point",
    "himmelblau",
]

PROBLEM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "quadratic": {
        "name": "Quadratic",
        "factory": lambda: QuadraticLoss(a=1.0, b=4.0, c=0.5, d=-2.0, e=-8.0, f=5.0),
        "start": (3.0, 3.0),
        "lr": {"SGD": 0.02, "Adam": 0.1, "Muon": 0.03},
        "bounds": {"x": [-2, 6], "y": [-2, 6]},
    },
    "ill_conditioned": {
        "name": "Ill-Conditioned (κ=1000)",
        "factory": lambda: IllConditionedQuadratic(condition_number=1000.0),
        "start": (10.0, 10.0),
        "lr": {"SGD": 0.001, "Adam": 0.01, "Muon": 0.005},
        "bounds": {"x": [-12, 12], "y": [-12, 12]},
    },
    "rosenbrock": {
        "name": "Rosenbrock",
        "factory": lambda: RosenbrockFunction(a=100.0),
        "start": (-1.0, 1.0),
        "lr": {"SGD": 0.001, "Adam": 0.01, "Muon": 0.005},
        "bounds": {"x": [-2, 2], "y": [-1, 3]},
    },
    "beale": {
        "name": "Beale",
        "factory": BealeFunction,
        "start": (1.0, 1.0),
        "lr": {"SGD": 0.001, "Adam": 0.01, "Muon": 0.005},
        "bounds": {"x": [-4, 4], "y": [-3, 3]},
    },
    "saddle_point": {
        "name": "Saddle Point",
        "factory": SaddlePoint,
        "start": (1.0, 1.0),
        "lr": {"SGD": 0.01, "Adam": 0.05, "Muon": 0.02},
        "bounds": {"x": [-3, 3], "y": [-3, 3]},
    },
    "himmelblau": {
        "name": "Himmelblau",
        "factory": HimmelblauFunction,
        "start": (0.0, 0.0),
        "lr": {"SGD": 0.005, "Adam": 0.02, "Muon": 0.01},
        "bounds": {"x": [-6, 6], "y": [-6, 6]},
    },
}


def _loss_at(problem: OptimizationProblem, x: float, y: float) -> float:
    return float(problem.compute_loss(torch.tensor(x), torch.tensor(y)).item())


def _compute_grid(
    problem: OptimizationProblem,
    bounds: Dict[str, List[float]],
    size: int = GRID_SIZE,
) -> Dict[str, Any]:
    x_min, x_max = bounds["x"]
    y_min, y_max = bounds["y"]
    xs = np.linspace(x_min, x_max, size).tolist()
    ys = np.linspace(y_min, y_max, size).tolist()
    values: List[List[float]] = []
    z_min = float("inf")
    z_max = float("-inf")
    for y in ys:
        row: List[float] = []
        for x in xs:
            z = _loss_at(problem, x, y)
            row.append(z)
            z_min = min(z_min, z)
            z_max = max(z_max, z)
        values.append(row)
    return {"x": xs, "y": ys, "values": values, "z_min": z_min, "z_max": z_max}


def _collect_trajectory(
    optimizer_name: str,
    problem: OptimizationProblem,
    initial_point: Tuple[float, float],
    learning_rate: float,
    steps: int,
) -> List[Dict[str, float]]:
    points: List[Dict[str, float]] = []

    if optimizer_name == "Muon":
        params = torch.tensor([initial_point], requires_grad=True, dtype=torch.float32)
        optimizer = Muon([params], lr=learning_rate, momentum=0.9)

        def loss_fn() -> torch.Tensor:
            return problem(params.view(-1))

        def get_pos() -> Tuple[float, float]:
            return params[0, 0].item(), params[0, 1].item()
    else:
        params = torch.tensor(initial_point, requires_grad=True, dtype=torch.float32)
        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD([params], lr=learning_rate)
        else:
            optimizer = torch.optim.Adam([params], lr=learning_rate)

        def loss_fn() -> torch.Tensor:
            return problem(params)

        def get_pos() -> Tuple[float, float]:
            return params[0].item(), params[1].item()

    x, y = get_pos()
    loss_val = loss_fn().item()
    points.append({"step": 0, "x": x, "y": y, "loss": float(loss_val)})

    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        x, y = get_pos()
        points.append({"step": step + 1, "x": x, "y": y, "loss": float(loss.item())})

    return points


def export_landscapes(profile: str, landscape_steps: int) -> None:
    problem_ids = LITE_PROBLEMS if profile == "lite" else FULL_PROBLEMS
    landscapes_dir = OUTPUT_DIR / "landscapes"
    landscapes_dir.mkdir(parents=True, exist_ok=True)

    index_entries: List[Dict[str, Any]] = []

    for pid in problem_ids:
        cfg = PROBLEM_CONFIGS[pid]
        problem = cfg["factory"]()
        bounds = cfg["bounds"]
        grid = _compute_grid(problem, bounds)
        trajectories: Dict[str, List[Dict[str, float]]] = {}
        for opt_name, lr in cfg["lr"].items():
            trajectories[opt_name] = _collect_trajectory(
                opt_name,
                problem,
                cfg["start"],
                lr,
                landscape_steps,
            )

        payload = {
            "id": pid,
            "name": cfg["name"],
            "description": f"2D benchmark on the {cfg['name']} loss landscape.",
            "minimum": {
                "x": problem.x_min,
                "y": problem.y_min,
                "loss": problem.f_min,
            },
            "initial_point": {"x": cfg["start"][0], "y": cfg["start"][1]},
            "learning_rates": cfg["lr"],
            "steps": landscape_steps,
            "bounds": bounds,
            "grid": grid,
            "trajectories": trajectories,
        }

        out_path = landscapes_dir / f"{pid}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        index_entries.append(
            {
                "id": pid,
                "name": cfg["name"],
                "description": payload["description"],
                "steps": landscape_steps,
            }
        )

    (landscapes_dir / "index.json").write_text(
        json.dumps({"profile": profile, "problems": index_entries}, indent=2),
        encoding="utf-8",
    )


def export_mnist(profile: str, mnist_epochs: int) -> None:
    device = torch.device("cpu")
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
            "kwargs": {"lr": 0.005, "momentum": 0.9, "weight_decay": 1e-4, "steps": 5},
        },
    ]

    optimizers_out: List[Dict[str, Any]] = []
    for config in optimizer_configs:
        tracker: OptimizationTracker = train_with_optimizer(
            config["name"],
            config["class"],
            config["kwargs"],
            epochs=mnist_epochs,
            device=device,
        )
        metrics = tracker.get_final_metrics()
        optimizers_out.append(
            {
                "name": tracker.name,
                "train_losses": tracker.train_losses,
                "train_accuracies": tracker.train_accuracies,
                "test_accuracies": tracker.test_accuracies,
                "epoch_times": tracker.epoch_times,
                "metrics": {
                    "final_test_acc": metrics["final_test_acc"],
                    "best_test_acc": metrics["best_test_acc"],
                    "final_train_loss": metrics["final_train_loss"],
                    "convergence_epoch": metrics["convergence_epoch"],
                    "avg_epoch_time": metrics["avg_epoch_time"],
                    "total_time": metrics["total_time"],
                },
            }
        )

    payload = {
        "profile": profile,
        "epochs": mnist_epochs,
        "batch_size": 128,
        "dataset": "MNIST",
        "optimizers": optimizers_out,
    }
    (OUTPUT_DIR / "mnist.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_definitions() -> None:
    definitions = {
        "metrics": {
            "best_test_acc": {
                "plain": "Highest test-set classification accuracy reached during training (%).",
            },
            "final_test_acc": {
                "plain": "Test accuracy after the final epoch (%).",
            },
            "convergence_epoch": {
                "plain": "First epoch where test accuracy reached 95% of the run's best accuracy.",
            },
            "avg_epoch_time": {"plain": "Mean wall-clock time per training epoch (seconds)."},
            "final_train_loss": {"plain": "Average training NLL loss on the final epoch."},
            "total_time": {"plain": "Total training wall-clock time (seconds)."},
        },
        "primer": {
            "muon": (
                "Muon applies SGD momentum, then replaces each 2D parameter update with "
                "its nearest orthogonal matrix via Newton-Schulz (or Polar Express) iteration."
            ),
            "hybrid": (
                "Matrix weights (2D+) use Muon; biases, embeddings, and output heads typically use AdamW."
            ),
        },
        "optimizers": {
            "SGD": {"plain": "Stochastic gradient descent with momentum."},
            "Adam": {"plain": "Adaptive moment estimation."},
            "Muon": {"plain": "MomentUm Orthogonalized by Newton-schulz."},
        },
    }
    (OUTPUT_DIR / "definitions.json").write_text(
        json.dumps(definitions, indent=2), encoding="utf-8"
    )


def export_index(profile: str) -> None:
    index = {
        "profile": profile,
        "suites": [
            {
                "id": "mnist",
                "name": "MNIST CNN",
                "description": "Small CNN on MNIST comparing SGD, Adam, and Muon.",
                "data_file": "mnist.json",
            },
            {
                "id": "landscapes",
                "name": "2D Landscapes",
                "description": "Classic 2D optimization benchmarks with trajectory overlays.",
                "data_file": "landscapes/index.json",
            },
        ],
    }
    (OUTPUT_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export explorer JSON data bundles.")
    parser.add_argument(
        "--profile",
        choices=["lite", "full"],
        default="lite",
        help="lite: fast export for CI; full: complete benchmarks",
    )
    parser.add_argument("--mnist-epochs", type=int, default=None)
    parser.add_argument("--landscape-steps", type=int, default=None)
    args = parser.parse_args()

    if args.profile == "lite":
        mnist_epochs = args.mnist_epochs if args.mnist_epochs is not None else 3
        landscape_steps = args.landscape_steps if args.landscape_steps is not None else 25
    else:
        mnist_epochs = args.mnist_epochs if args.mnist_epochs is not None else 15
        landscape_steps = args.landscape_steps if args.landscape_steps is not None else 30

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting profile={args.profile} to {OUTPUT_DIR}")
    export_definitions()
    export_mnist(args.profile, mnist_epochs)
    export_landscapes(args.profile, landscape_steps)
    export_index(args.profile)
    print("Done.")


if __name__ == "__main__":
    main()
