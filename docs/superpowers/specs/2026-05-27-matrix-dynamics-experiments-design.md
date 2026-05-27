# Design Spec: Matrix Dynamics Experiments & Explorer Integration

**Date:** 2026-05-27  
**Status:** Approved  
**Topic:** Visualizing the effect of Muon optimization on 2D Matrix Parameters via Deep Matrix Factorization and MNIST Vision Transformers.

---

## 1. Goal & Context
The Muon optimizer performs orthogonalized momentum updates for 2D matrix parameters, preventing singular value decay (rank collapse) and keeping matrices well-conditioned. This design specifies two new experiment scripts (Option A: Deep Matrix Factorization; Option B: MNIST Vision Transformer) and their integration into the existing Vite-based Explorer dashboard.

---

## 2. Technical Architecture & Scripts

### 2.1 Deep Matrix Factorization (`matrix_factorization_benchmark.py`)
This script compares SGD, AdamW, and Muon in optimizing a 3-layer deep linear network.

* **Target Function:** We learn a fixed matrix projection $Y = M_{\text{target}} X$.
  * Input $X \in \mathbb{R}^{64 \times 256}$ drawn from $\mathcal{N}(0, 1)$.
  * $M_{\text{target}} \in \mathbb{R}^{64 \times 64}$ is pre-computed with exponentially decaying singular values: $\sigma_i = 10^{-2i/63}$ for $i=0 \dots 63$ (minimum singular value is $0.01$, condition number $\kappa = 100$).
* **Network Model:** $\hat{Y} = W_3 W_2 W_1 X$, where $W_1, W_2, W_3 \in \mathbb{R}^{64 \times 64}$.
* **Optimizers:**
  * **SGD:** Learning rate = 0.05, momentum = 0.95.
  * **AdamW:** Learning rate = 1e-3, betas = (0.9, 0.95).
  * **SingleDeviceMuon:** Learning rate = 0.02, momentum = 0.95.
* **Interval measurements:** Every 10 steps, record MSE loss. At steps $0, 50, 100, 250, 500$, compute and save the complete set of 64 singular values for $W_1, W_2, W_3$.

---

### 2.2 MNIST Vision Transformer (`transformer_spectral_benchmark.py`)
This script compares pure AdamW and Muon + Aux AdamW when training a small Vision Transformer on MNIST.

* **ViT Architecture:**
  * Patch size: $4 \times 4$ pixels (resulting in a sequence length of 49 tokens).
  * Embedding Dimension: 64.
  * Attention Blocks: 2 layers, 4 heads. Query, Key, Value, Output projection matrices are size $64 \times 64$.
  * MLP block expansion: $64 \to 128 \to 64$.
  * Classifier: Classification token output projected $64 \to 10$ classes.
* **Optimizers Compared:**
  * **Pure AdamW:** LR = 1e-3, weight decay = 0.01, betas = (0.9, 0.95).
  * **Muon + Aux AdamW:** Muon LR = 0.02 for 2D weights; AdamW LR = 3e-4 for patch projections, embeddings, LayerNorms, and biases.
* **Interval measurements:** Log training loss and validation accuracy. At steps $0, 100, 250, 500$, calculate and record:
  1. Singular values of the Q & K attention projection weights in layer 1.
  2. Average entropy of attention weights across all heads.

---

## 3. Data Export & Serialization
Both scripts will export their results into JSON files located under `explorer/public/data/`:
* `explorer/public/data/matrix_factorization.json`
* `explorer/public/data/transformer_spectral.json`

The JSON structure maps directly to Recharts-friendly data series:
```json
{
  "steps": [0, 10, 20, ..., 500],
  "optimizers": [
    {
      "name": "SGD",
      "losses": [1.2, 0.9, ...],
      "condition_numbers": [1.0, 1.5, ...],
      "singular_values": {
        "0": [1.0, 0.99, ...],
        "100": [0.8, 0.7, ...],
        "500": [0.1, 0.05, ...]
      }
    }
  ]
}
```

We will also update `scripts/export_explorer_data.py` to trigger these runs during the static data export flow.

---

## 4. Explorer Dashboard Changes

We will introduce a new explorer interface:
1. **Header Tab:** Add a `Matrix Dynamics` button in `App.tsx` mapped to a new sub-explorer component `explorer/src/matrix/MatrixDynamicsExplorer.tsx`.
2. **Visual Components:**
   * **Comparison Charts:** Line graphs tracking loss trajectories and condition numbers over steps.
   * **Spectrum Histogram/Bar Chart:** Bar chart representing singular values. An interactive step-selection tab allows viewing the spectrum at Step 0, 100, 250, or 500, demonstrating Muon's flat/uniform singular value profile versus SGD/AdamW's skewed profile.
   * **Attention Entropy Chart:** Line graph comparing attention entropy dynamics (Muon vs. AdamW).

---

## 5. Verification Plan

* **Validation Scripts:** Run `uv run python matrix_factorization_benchmark.py` and `uv run python transformer_spectral_benchmark.py` locally and verify terminal console formatting and image output.
* **Data Verification:** Confirm that JSON output files are formatted correctly and generated under `explorer/public/data/`.
* **UI Verification:** Launch the Vite server (`npm run dev` in `explorer/`), navigate to the new tab, and verify that the plots render correctly. Ensure TypeScript compiles without errors.
