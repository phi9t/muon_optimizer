# Matrix Dynamics Experiments & Explorer UI Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement two new optimization benchmark scripts (deep matrix factorization and MNIST Vision Transformer) to showcase Muon's advantages for 2D matrix parameters, and integrate their results dynamically into the web explorer dashboard.

**Architecture:** We will build stand-alone benchmarking scripts that execute the runs and serialize the results (losses, singular values, condition numbers, entropy) as static JSON files. The Vite dashboard will be expanded with a React component using Recharts to visualize these files.

**Tech Stack:** Python 3.12, PyTorch, Recharts, Vite, React (TypeScript).

---

### Task 1: Deep Matrix Factorization Benchmark Script

**Files:**
- Create: `matrix_factorization_benchmark.py`
- Test: `matrix_factorization_benchmark_test.py`

- [ ] **Step 1: Write a basic test for synthetic data generation and matrix decomposition**
  Create `matrix_factorization_benchmark_test.py`:
  ```python
  import torch
  import numpy as np
  from matrix_factorization_benchmark import make_ill_conditioned_matrix, run_mf_experiment

  def test_make_ill_conditioned_matrix():
      M = make_ill_conditioned_matrix(64)
      assert M.shape == (64, 64)
      U, S, Vh = torch.linalg.svd(M)
      assert torch.allclose(S[0], torch.tensor(1.0), atol=1e-2)
      assert torch.allclose(S[-1], torch.tensor(0.01), atol=1e-2)
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `uv run pytest matrix_factorization_benchmark_test.py -v`
  Expected: FAIL with ModuleNotFoundError or import error.

- [ ] **Step 3: Implement matrix generation and core experimental loop**
  Create `matrix_factorization_benchmark.py` containing the logic to run the experiment:
  ```python
  import json
  import torch
  import torch.nn as nn
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

  def run_mf_experiment():
      dim = 64
      steps = 500
      M_target = make_ill_conditioned_matrix(dim)
      X = torch.randn(dim, 256)
      Y = M_target @ X

      results = {}
      for opt_name in ["SGD", "AdamW", "Muon"]:
          W1 = torch.randn(dim, dim, requires_grad=True) * 0.1
          W2 = torch.randn(dim, dim, requires_grad=True) * 0.1
          W3 = torch.randn(dim, dim, requires_grad=True) * 0.1
          
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
                  
                  if step in [0, 50, 100, 250, 499]:
                      sv_snapshots[str(step)] = s_vals.tolist()

          results[opt_name] = {
              "losses": losses,
              "cond_numbers": cond_nums,
              "singular_values": sv_snapshots
          }

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
      plt.legend()

      plt.subplot(1, 2, 2)
      for name, res in results.items():
          plt.plot(res["singular_values"]["499"], label=name)
      plt.title("Singular Value Spectrum at Step 500")
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
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `uv run pytest matrix_factorization_benchmark_test.py -v`
  Expected: PASS.

- [ ] **Step 5: Run the experiment CLI script once**
  Run: `uv run python matrix_factorization_benchmark.py`
  Expected: Prints table and writes files to `explorer/public/data/matrix_factorization.json` and `assets/matrix_factorization_benchmark.png`.

- [ ] **Step 6: Commit**
  ```bash
  git add matrix_factorization_benchmark.py matrix_factorization_benchmark_test.py
  git commit -m "feat: implement deep matrix factorization experiment and testing"
  ```

---

### Task 2: MNIST Vision Transformer Script

**Files:**
- Create: `transformer_spectral_benchmark.py`
- Test: `transformer_spectral_benchmark_test.py`

- [ ] **Step 2.1: Write failing test verifying ViT module creation and parameter count**
  Create `transformer_spectral_benchmark_test.py`:
  ```python
  import torch
  from transformer_spectral_benchmark import SimpleViT

  def test_vit_output_shape():
      model = SimpleViT(image_size=28, patch_size=4, num_classes=10, embed_dim=64, depth=2, heads=4)
      x = torch.randn(8, 1, 28, 28)
      out = model(x)
      assert out.shape == (8, 10)
  ```

- [ ] **Step 2.2: Run test to verify it fails**
  Run: `uv run pytest transformer_spectral_benchmark_test.py -v`
  Expected: FAIL with import error.

- [ ] **Step 2.3: Implement SimpleViT architecture and training run**
  Create `transformer_spectral_benchmark.py`:
  ```python
  import json
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from torch.utils.data import DataLoader
  from torchvision import datasets, transforms
  from pathlib import Path
  import matplotlib.pyplot as plt
  from rich.console import Console
  from rich.table import Table
  from muon_optimizer import SingleDeviceMuonWithAuxAdam, create_muon_param_groups

  class Attention(nn.Module):
      def __init__(self, dim, heads=4):
          super().__init__()
          self.heads = heads
          self.scale = (dim // heads) ** -0.5
          self.qkv = nn.Linear(dim, dim * 3, bias=False)
          self.proj = nn.Linear(dim, dim, bias=False)

      def forward(self, x):
          B, N, C = x.shape
          qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, B, 3, 1, 4)
          q, k, v = qkv[0], qkv[1], qkv[2]
          attn = (q @ k.transpose(-2, -1)) * self.scale
          attn = attn.softmax(dim=-1)
          out = (attn @ v).transpose(1, 2).reshape(B, N, C)
          return self.proj(out), attn

  class TransformerBlock(nn.Module):
      def __init__(self, dim, heads=4):
          super().__init__()
          self.norm1 = nn.LayerNorm(dim)
          self.attn = Attention(dim, heads)
          self.norm2 = nn.LayerNorm(dim)
          self.mlp = nn.Sequential(
              nn.Linear(dim, dim * 2),
              nn.GELU(),
              nn.Linear(dim * 2, dim)
          )

      def forward(self, x):
          h, attn_map = self.attn(self.norm1(x))
          x = x + h
          x = x + self.mlp(self.norm2(x))
          return x, attn_map

  class SimpleViT(nn.Module):
      def __init__(self, image_size=28, patch_size=4, num_classes=10, embed_dim=64, depth=2, heads=4):
          super().__init__()
          num_patches = (image_size // patch_size) ** 2
          self.patch_size = patch_size
          self.patch_to_embed = nn.Linear(patch_size * patch_size, embed_dim)
          self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
          self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
          self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)])
          self.norm = nn.LayerNorm(embed_dim)
          self.classifier = nn.Linear(embed_dim, num_classes)

      def forward(self, x):
          # Patchify
          p = self.patch_size
          B, C, H, W = x.shape
          x = x.unfold(2, p, p).unfold(3, p, p) # B, C, H/p, W/p, p, p
          x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, p*p)
          
          x = self.patch_to_embed(x)
          cls_tokens = self.cls_token.expand(B, -1, -1)
          x = torch.cat((cls_tokens, x), dim=1)
          x = x + self.pos_embed

          attn_maps = []
          for block in self.blocks:
              x, attn = block(x)
              attn_maps.append(attn)
              
          x = self.norm(x[:, 0])
          return self.classifier(x), attn_maps

  def compute_entropy(attn):
      # attn shape: [B, heads, N, N]
      entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1)
      return float(entropy.mean().item())

  def run_vit_experiment():
      device = torch.device("cpu")
      transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
      train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
      # Subset of 2000 images for fast CPU benchmark run
      subset = torch.utils.data.Subset(train_dataset, range(2000))
      train_loader = DataLoader(subset, batch_size=64, shuffle=True)

      results = {}
      for opt_name in ["AdamW", "Muon+Aux"]:
          model = SimpleViT().to(device)
          
          if opt_name == "AdamW":
              opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
          else:
              groups = create_muon_param_groups(model, muon_lr=0.02, adam_lr=3e-4, weight_decay=0.01)
              opt = SingleDeviceMuonWithAuxAdam(groups)

          losses = []
          accuracies = []
          entropies = []
          sv_snapshots = {}

          step = 0
          for epoch in range(2): # 2 short epochs
              for data, target in train_loader:
                  data, target = data.to(device), target.to(device)
                  opt.zero_grad()
                  output, attns = model(data)
                  loss = F.cross_entropy(output, target)
                  loss.backward()
                  opt.step()

                  # Evaluate accuracy on current batch
                  pred = output.argmax(dim=1)
                  acc = (pred == target).float().mean().item()
                  
                  losses.append(float(loss.item()))
                  accuracies.append(acc)
                  entropies.append(compute_entropy(attns[0]))

                  # Record Q projection singular values
                  with torch.no_grad():
                      # Look at self-attention Q projection weight parameter
                      qkv_weight = model.blocks[0].attn.qkv.weight
                      q_weight = qkv_weight[:64] # Slice query weights
                      s_vals = torch.linalg.svdvals(q_weight)
                      if step in [0, 10, 30, 50]:
                          sv_snapshots[str(step)] = s_vals.tolist()

                  step += 1
                  if step >= 60:
                      break
              if step >= 60:
                  break

          results[opt_name] = {
              "losses": losses,
              "accuracies": accuracies,
              "entropies": entropies,
              "singular_values": sv_snapshots
          }

      # Write out JSON
      out_dir = Path("explorer/public/data")
      out_dir.mkdir(parents=True, exist_ok=True)
      with open(out_dir / "transformer_spectral.json", "w") as f:
          json.dump(results, f, indent=2)

      # Create plotting
      plt.figure(figsize=(10, 4))
      plt.subplot(1, 2, 1)
      for name, res in results.items():
          plt.plot(res["losses"], label=name)
      plt.title("ViT MNIST Training Loss")
      plt.legend()
      plt.subplot(1, 2, 2)
      for name, res in results.items():
          last_step = max(res["singular_values"].keys(), key=int)
          plt.plot(res["singular_values"][last_step], label=name)
      plt.title("Q-Projection Singular Values")
      plt.legend()
      plt.savefig("assets/transformer_spectral_benchmark.png")
      plt.close()

      console = Console()
      table = Table(title="ViT Optimization Summary")
      table.add_column("Optimizer", style="cyan")
      table.add_column("Final Loss", style="green")
      table.add_column("Final Head Entropy", style="magenta")
      for name, res in results.items():
          table.add_row(name, f"{res['losses'][-1]:.4f}", f"{res['entropies'][-1]:.4f}")
      console.print(table)

  if __name__ == "__main__":
      run_vit_experiment()
  ```

- [ ] **Step 2.4: Run test to verify it passes**
  Run: `uv run pytest transformer_spectral_benchmark_test.py -v`
  Expected: PASS.

- [ ] **Step 2.5: Execute the benchmark script once**
  Run: `uv run python transformer_spectral_benchmark.py`
  Expected: Prints table and writes files to `explorer/public/data/transformer_spectral.json` and `assets/transformer_spectral_benchmark.png`.

- [ ] **Step 2.6: Commit**
  ```bash
  git add transformer_spectral_benchmark.py transformer_spectral_benchmark_test.py
  git commit -m "feat: implement Vision Transformer MNIST spectral benchmark"
  ```

---

### Task 3: Integration into export script

**Files:**
- Modify: `scripts/export_explorer_data.py`

- [ ] **Step 3.1: Incorporate calls to the two new experiments into the export data flow**
  Add imports and function calls under `scripts/export_explorer_data.py`. Let's view the imports and append run commands.
  In `scripts/export_explorer_data.py`:
  ```python
  from matrix_factorization_benchmark import run_mf_experiment
  from transformer_spectral_benchmark import run_vit_experiment
  ```
  And inside `main` function (around line 340-360), call them:
  ```python
  print("Running Matrix Factorization experiment...")
  run_mf_experiment()
  print("Running ViT MNIST experiment...")
  run_vit_experiment()
  ```

- [ ] **Step 3.2: Verify running export script executes them**
  Run: `uv run python scripts/export_explorer_data.py --profile lite`
  Expected: Runs successfully, producing all files without errors.

- [ ] **Step 3.3: Commit changes**
  ```bash
  git add scripts/export_explorer_data.py
  git commit -m "feat: integrate matrix dynamic benchmarks with export_explorer_data.py"
  ```

---

### Task 4: Explorer UI - Component implementation

**Files:**
- Create: `explorer/src/matrix/MatrixDynamicsExplorer.tsx`
- Modify: `explorer/src/App.tsx`

- [ ] **Step 4.1: Implement the React component `MatrixDynamicsExplorer.tsx`**
  Write this file using Recharts to present line charts of losses, condition numbers, and singular value distributions.
  Create `explorer/src/matrix/MatrixDynamicsExplorer.tsx`:
  ```tsx
  import { useEffect, useState } from 'react'
  import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    BarChart,
    Bar,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
  } from 'recharts'
  import { errorMessage, fetchExplorerJson } from '../lib/fetch'

  const COLORS: Record<string, string> = {
    SGD: '#ef4444',
    AdamW: '#3b82f6',
    Muon: '#10b981',
    'Muon+Aux': '#10b981',
  }

  export default function MatrixDynamicsExplorer() {
    const [subTab, setSubTab] = useState<'mf' | 'vit'>('mf')
    const [mfData, setMfData] = useState<any>(null)
    const [vitData, setVitData] = useState<any>(null)
    const [stepSel, setStepSel] = useState<string>('499')
    const [vitStepSel, setVitStepSel] = useState<string>('50')
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
      async function load() {
        try {
          setLoading(true)
          const [mf, vit] = await Promise.all([
            fetchExplorerJson<any>('matrix_factorization.json'),
            fetchExplorerJson<any>('transformer_spectral.json'),
          ])
          setMfData(mf)
          setVitData(vit)
        } catch (err: any) {
          setError(errorMessage(err))
        } finally {
          setLoading(false)
        }
      }
      void load()
    }, [])

    if (loading) return <div className="loading-container">Loading Matrix Dynamics...</div>
    if (error) return <div className="text-danger">Error: {error}</div>

    // Build charts data
    const getLossData = (data: any) => {
      if (!data) return []
      const len = data[Object.keys(data)[0]].losses.length
      const rows = []
      for (let i = 0; i < len; i++) {
        const row: any = { step: i }
        for (const name of Object.keys(data)) {
          row[name] = data[name].losses[i]
        }
        rows.push(row)
      }
      return rows
    }

    const getCondData = (data: any) => {
      if (!data) return []
      const len = data[Object.keys(data)[0]].cond_numbers.length
      const rows = []
      for (let i = 0; i < len; i++) {
        const row: any = { step: i }
        for (const name of Object.keys(data)) {
          row[name] = data[name].cond_numbers[i]
        }
        rows.push(row)
      }
      return rows
    }

    const getSpectrumData = (data: any, stepKey: string) => {
      if (!data) return []
      const optimizers = Object.keys(data)
      const listLen = data[optimizers[0]].singular_values[stepKey]?.length || 0
      const rows = []
      for (let i = 0; i < listLen; i++) {
        const row: any = { index: i }
        for (const name of optimizers) {
          row[name] = data[name].singular_values[stepKey]?.[i] || 0
        }
        rows.push(row)
      }
      return rows
    }

    const getEntropyData = (data: any) => {
      if (!data) return []
      const len = data[Object.keys(data)[0]].entropies.length
      const rows = []
      for (let i = 0; i < len; i++) {
        const row: any = { step: i }
        for (const name of Object.keys(data)) {
          row[name] = data[name].entropies[i]
        }
        rows.push(row)
      }
      return rows
    }

    return (
      <div className="mo-shell pb-shell">
        <div className="family-switch sub-tab-switch" style={{ marginBottom: '1.5rem' }}>
          <button
            type="button"
            className={`family-switch-btn ${subTab === 'mf' ? 'active' : ''}`}
            onClick={() => setSubTab('mf')}
          >
            Deep Matrix Factorization
          </button>
          <button
            type="button"
            className={`family-switch-btn ${subTab === 'vit' ? 'active' : ''}`}
            onClick={() => setSubTab('vit')}
          >
            MNIST Vision Transformer
          </button>
        </div>

        {subTab === 'mf' ? (
          <div>
            <h3>Deep Matrix Factorization (Linear Network)</h3>
            <p className="text-secondary" style={{ marginBottom: '1.5rem' }}>
              Optimizing a 3-layer deep linear structure $W_3 W_2 W_1 X$ to recover an ill-conditioned target matrix.
            </p>
            <div className="chart-grid">
              <div className="chart-card">
                <h4>Training MSE Loss Trajectory</h4>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={getLossData(mfData)}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="step" stroke="#9ca3af" fontSize={11} />
                    <YAxis stroke="#9ca3af" fontSize={11} scale="log" domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.1)' }} />
                    <Legend />
                    {Object.keys(mfData).map((opt) => (
                      <Line key={opt} type="monotone" dataKey={opt} stroke={COLORS[opt]} dot={false} strokeWidth={2} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-card">
                <h4>Condition Number $\kappa(W_2)$</h4>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={getCondData(mfData)}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="step" stroke="#9ca3af" fontSize={11} />
                    <YAxis stroke="#9ca3af" fontSize={11} />
                    <Tooltip contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.1)' }} />
                    <Legend />
                    {Object.keys(mfData).map((opt) => (
                      <Line key={opt} type="monotone" dataKey={opt} stroke={COLORS[opt]} dot={false} strokeWidth={2} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="chart-card" style={{ marginTop: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h4>Singular Value Spectrum Profile</h4>
                <div>
                  <label htmlFor="step-select" style={{ marginRight: '0.5rem', fontSize: '12px' }}>Training Step: </label>
                  <select
                    id="step-select"
                    value={stepSel}
                    onChange={(e) => setStepSel(e.target.value)}
                    style={{ background: '#1f2937', color: '#f3f4f6', border: '1px solid #4b5563', borderRadius: '4px', padding: '2px 8px' }}
                  >
                    <option value="0">Step 0 (Init)</option>
                    <option value="50">Step 50</option>
                    <option value="100">Step 100</option>
                    <option value="250">Step 250</option>
                    <option value="499">Step 500 (Final)</option>
                  </select>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={getSpectrumData(mfData, stepSel)}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="index" stroke="#9ca3af" fontSize={10} />
                  <YAxis stroke="#9ca3af" fontSize={11} />
                  <Tooltip contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.1)' }} />
                  <Legend />
                  {Object.keys(mfData).map((opt) => (
                    <Bar key={opt} dataKey={opt} fill={COLORS[opt]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          <div>
            <h3>MNIST Vision Transformer (ViT)</h3>
            <p className="text-secondary" style={{ marginBottom: '1.5rem' }}>
              Training a 2-layer ViT on MNIST classification comparing pure AdamW and Hybrid Muon + Aux AdamW.
            </p>
            <div className="chart-grid">
              <div className="chart-card">
                <h4>Training Loss Trajectory</h4>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={getLossData(vitData)}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="step" stroke="#9ca3af" fontSize={11} />
                    <YAxis stroke="#9ca3af" fontSize={11} />
                    <Tooltip contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.1)' }} />
                    <Legend />
                    {Object.keys(vitData).map((opt) => (
                      <Line key={opt} type="monotone" dataKey={opt} stroke={COLORS[opt]} dot={false} strokeWidth={2} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-card">
                <h4>Attention Map Entropy</h4>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={getEntropyData(vitData)}>
                    <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="step" stroke="#9ca3af" fontSize={11} />
                    <YAxis stroke="#9ca3af" fontSize={11} />
                    <Tooltip contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.1)' }} />
                    <Legend />
                    {Object.keys(vitData).map((opt) => (
                      <Line key={opt} type="monotone" dataKey={opt} stroke={COLORS[opt]} dot={false} strokeWidth={2} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="chart-card" style={{ marginTop: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h4>Query Projection Singular Value Spectrum</h4>
                <div>
                  <label htmlFor="vit-step-select" style={{ marginRight: '0.5rem', fontSize: '12px' }}>Training Step: </label>
                  <select
                    id="vit-step-select"
                    value={vitStepSel}
                    onChange={(e) => setVitStepSel(e.target.value)}
                    style={{ background: '#1f2937', color: '#f3f4f6', border: '1px solid #4b5563', borderRadius: '4px', padding: '2px 8px' }}
                  >
                    <option value="0">Step 0 (Init)</option>
                    <option value="10">Step 10</option>
                    <option value="30">Step 30</option>
                    <option value="50">Step 50 (Final)</option>
                  </select>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={getSpectrumData(vitData, vitStepSel)}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="index" stroke="#9ca3af" fontSize={10} />
                  <YAxis stroke="#9ca3af" fontSize={11} />
                  <Tooltip contentStyle={{ background: '#111827', border: '1px solid rgba(255,255,255,0.1)' }} />
                  <Legend />
                  {Object.keys(vitData).map((opt) => (
                    <Bar key={opt} dataKey={opt} fill={COLORS[opt]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    )
  }
  ```

- [ ] **Step 4.2: Update Navigation in `explorer/src/App.tsx`**
  Modify `App.tsx` to support the new section:
  Add tab key `'matrix'` to `ExplorerFamily` enum and `FAMILY_SUBTITLES` dictionary:
  ```typescript
  export type ExplorerFamily = 'benchmarks' | 'landscapes' | 'reference' | 'qwen' | 'matrix'
  ```
  ```typescript
  matrix: 'Singular value and matrix dynamics on deep linear networks and Vision Transformers',
  ```
  Add a button for "Matrix Dynamics" in the `family-switch` header:
  ```tsx
  <button
    type="button"
    className={`family-switch-btn ${family === 'matrix' ? 'active' : ''}`}
    onClick={() => setFamily('matrix')}
  >
    Matrix Dynamics
  </button>
  ```
  Render component in the body:
  ```tsx
  {family === 'matrix' && <MatrixDynamicsExplorer />}
  ```
  Import component at the top:
  ```typescript
  import MatrixDynamicsExplorer from './matrix/MatrixDynamicsExplorer'
  ```

- [ ] **Step 4.3: Verify Vite builds and typechecks successfully**
  Run: `cd explorer && npm run build`
  Expected: Successful compilation without warnings or errors.

- [ ] **Step 4.4: Commit changes**
  ```bash
  git add explorer/src/matrix/MatrixDynamicsExplorer.tsx explorer/src/App.tsx
  git commit -m "feat: implement MatrixDynamicsExplorer UI and App.tsx navigation"
  ```
