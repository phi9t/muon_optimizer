import torch
from matrix_factorization_benchmark import (
    make_ill_conditioned_matrix,
    run_mf_experiment_logic,
)

def test_make_ill_conditioned_matrix():
    M = make_ill_conditioned_matrix(64)
    assert M.shape == (64, 64)
    U, S, Vh = torch.linalg.svd(M)
    assert torch.allclose(S[0], torch.tensor(1.0), atol=1e-2)
    assert torch.allclose(S[-1], torch.tensor(0.01), atol=1e-2)

def test_mf_experiment_logic():
    dim = 8
    steps = 5
    torch.manual_seed(42)
    
    M_target = make_ill_conditioned_matrix(dim)
    X = torch.randn(dim, 16)
    Y = M_target @ X
    
    W1_init = torch.randn(dim, dim) * 0.1
    W2_init = torch.randn(dim, dim) * 0.1
    W3_init = torch.randn(dim, dim) * 0.1
    
    results = run_mf_experiment_logic(dim, steps, X, Y, W1_init, W2_init, W3_init)
    
    assert isinstance(results, dict)
    for opt_name in ["SGD", "AdamW", "Muon"]:
        assert opt_name in results
        opt_res = results[opt_name]
        assert "losses" in opt_res
        assert "cond_numbers" in opt_res
        assert "singular_values" in opt_res
        
        assert len(opt_res["losses"]) == steps
        assert len(opt_res["cond_numbers"]) == steps
        
        # At step 0 and step 4 (steps-1), it should capture singular values
        assert "0" in opt_res["singular_values"]
        assert "4" in opt_res["singular_values"]
        assert len(opt_res["singular_values"]["0"]) == dim
