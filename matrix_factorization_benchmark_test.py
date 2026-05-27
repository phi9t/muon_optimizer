import torch
import numpy as np
from matrix_factorization_benchmark import make_ill_conditioned_matrix, run_mf_experiment

def test_make_ill_conditioned_matrix():
    M = make_ill_conditioned_matrix(64)
    assert M.shape == (64, 64)
    U, S, Vh = torch.linalg.svd(M)
    assert torch.allclose(S[0], torch.tensor(1.0), atol=1e-2)
    assert torch.allclose(S[-1], torch.tensor(0.01), atol=1e-2)
