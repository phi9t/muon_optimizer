import json
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from muon_optimizer import SingleDeviceMuon

def make_ill_conditioned_matrix(dim: int = 64) -> torch.Tensor:
    # Generate exponentially decaying singular values
    s = torch.exp(torch.linspace(0, -4.605, steps=dim)) # e^-4.605 ~ 0.01
    U, _ = torch.linalg.qr(torch.randn(dim, dim))
    V, _ = torch.linalg.qr(torch.randn(dim, dim))
    return U @ torch.diag(s) @ V.T

def run_mf_experiment_logic(
    dim: int,
    steps: int,
    X: torch.Tensor,
    Y: torch.Tensor,
    W1_init: torch.Tensor,
    W2_init: torch.Tensor,
    W3_init: torch.Tensor,
) -> dict:
    results = {}
    for opt_name in ["SGD", "AdamW", "Muon"]:
        W1 = W1_init.clone().detach().requires_grad_(True)
        W2 = W2_init.clone().detach().requires_grad_(True)
        W3 = W3_init.clone().detach().requires_grad_(True)
        
        if opt_name == "SGD":
            opt = torch.optim.SGD([W1, W2, W3], lr=0.05, momentum=0.95)
        elif opt_name == "AdamW":
            opt = torch.optim.AdamW([W1, W2, W3], lr=1e-3, betas=(0.9, 0.95))
        else:
            opt = SingleDeviceMuon([W1, W2, W3], lr=0.02, momentum=0.95)

        losses = []
        cond_nums = []
        sv_snapshots = {}

        for step in range(steps):
            opt.zero_grad()
            y_pred = W3 @ W2 @ W1 @ X
            loss = torch.mean((Y - y_pred) ** 2)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            
            # Record condition numbers for W2 as a representative
            with torch.no_grad():
                s_vals = torch.linalg.svdvals(W2)
                cond = float((s_vals[0] / (s_vals[-1] + 1e-8)).item())
                cond_nums.append(cond)
                
                # Record snapshots at standard checkpoints and the final step
                if step in [0, 50, 100, 250, 499] or step == steps - 1:
                    sv_snapshots[str(step)] = s_vals.tolist()

        results[opt_name] = {
            "losses": losses,
            "cond_numbers": cond_nums,
            "singular_values": sv_snapshots
        }
    return results

def run_mf_experiment() -> None:
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    dim = 64
    steps = 500
    M_target = make_ill_conditioned_matrix(dim)
    X = torch.randn(dim, 256)
    Y = M_target @ X

    # Draw template weights outside the optimizer loop for fairness
    W1_init = torch.randn(dim, dim) * 0.1
    W2_init = torch.randn(dim, dim) * 0.1
    W3_init = torch.randn(dim, dim) * 0.1

    results = run_mf_experiment_logic(dim, steps, X, Y, W1_init, W2_init, W3_init)

    # Save JSON data
    out_dir = Path("explorer/public/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "matrix_factorization.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for name, res in results.items():
        plt.plot(res["losses"], label=name)
    plt.yscale("log")
    plt.title("MSE Loss Trajectory")
    plt.xlabel("Step")
    plt.ylabel("Loss (log scale)")
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, res in results.items():
        plt.plot(res["singular_values"]["499"], label=name)
    plt.title("Singular Value Spectrum at Step 500")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Singular Value")
    plt.legend()
    
    plt.tight_layout()
    Path("assets").mkdir(exist_ok=True)
    plt.savefig("assets/matrix_factorization_benchmark.png")
    plt.close()

    # Render console table
    console = Console()
    table = Table(title="Deep Matrix Factorization Summary")
    table.add_column("Optimizer", style="cyan")
    table.add_column("Final Loss", style="green")
    table.add_column("Final Condition Number", style="magenta")
    for name, res in results.items():
        table.add_row(name, f"{res['losses'][-1]:.6f}", f"{res['cond_numbers'][-1]:.2f}")
    console.print(table)

if __name__ == "__main__":
    run_mf_experiment()
