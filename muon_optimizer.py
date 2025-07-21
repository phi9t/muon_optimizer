"""
Muon Optimizer - A PyTorch optimizer that performs orthogonalized momentum updates.

This module implements the Muon optimizer as described in the paper:
"Muon: MomentUm Orthogonalized by Newton-schulz"

The optimizer combines standard SGD-momentum with an orthogonalization post-processing step,
where each 2D parameter's update is replaced with the nearest orthogonal matrix using
Newton-Schulz iteration for efficient computation.

Key Features:
- Orthogonalized momentum updates for better training stability
- Efficient Newton-Schulz iteration for orthogonalization
- Support for both distributed and single-device training
- Hybrid optimization with AdamW for non-compatible parameters
- Stable bfloat16 computation on GPU

Author: Based on implementation by Keller Jordan and contributors
License: MIT
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer

# Configure logging
logger = logging.getLogger(__name__)


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Compute the zeroth power / orthogonalization of G using Newton-Schulz iteration.

    This function implements a quintic Newton-Schulz iteration with coefficients
    optimized to maximize the slope at zero. The iteration produces an approximation
    of the orthogonal component of the input matrix.

    Args:
        G: Input tensor of shape (..., m, n) where m, n >= 2
        steps: Number of Newton-Schulz iteration steps (typically 5)

    Returns:
        Orthogonalized tensor of the same shape as G

    Raises:
        AssertionError: If G has fewer than 2 dimensions

    Note:
        The iteration produces US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
        which empirically doesn't hurt model performance compared to exact UV^T.

    References:
        - Original implementation by @scottjmaddox and @YouJiacheng
        - Quintic computation strategy adapted from @jxbz, @leloykun, and @YouJiacheng
    """
    if G.ndim < 2:
        raise ValueError(f"Input tensor must have at least 2 dimensions, got {G.ndim}")

    # Optimized coefficients for quintic iteration
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Convert to bfloat16 for stability and efficiency
    X = G.bfloat16()

    # Handle rectangular matrices by transposing if needed
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize to ensure spectral norm is at most 1
    norm_factor = X.norm(dim=(-2, -1), keepdim=True) + 1e-7
    X = X / norm_factor

    # Perform Newton-Schulz iterations
    for step in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # Quintic computation
        X = a * X + B @ X

    # Transpose back if we transposed earlier
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


def muon_update(
    grad: Tensor,
    momentum: Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> Tensor:
    """
    Perform a Muon update step combining momentum and orthogonalization.

    This function implements the core Muon update algorithm:
    1. Update momentum buffer using exponential moving average
    2. Apply Nesterov momentum if enabled
    3. Orthogonalize the update using Newton-Schulz iteration
    4. Scale the update based on matrix dimensions

    Args:
        grad: Gradient tensor
        momentum: Momentum buffer tensor (will be modified in-place)
        beta: Momentum coefficient (default: 0.95)
        ns_steps: Number of Newton-Schulz iteration steps (default: 5)
        nesterov: Whether to use Nesterov momentum (default: True)

    Returns:
        Orthogonalized update tensor

    Note:
        For convolutional filters (4D tensors), the tensor is reshaped to 2D
        before orthogonalization and then reshaped back.
    """
    # Update momentum buffer
    momentum.lerp_(grad, 1 - beta)

    # Apply Nesterov momentum if enabled
    if nesterov:
        update = grad.lerp_(momentum, beta)
    else:
        update = momentum

    # Handle 1D parameters (like biases) - skip orthogonalization
    if update.ndim == 1:
        return update

    # Handle convolutional filters by reshaping to 2D
    original_shape = update.shape
    if update.ndim == 4:
        update = update.view(len(update), -1)

    # Orthogonalize the update
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    # Scale based on matrix dimensions
    scale_factor = max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    update *= scale_factor

    # Reshape back to original shape if needed
    if len(original_shape) == 4:
        update = update.view(original_shape)

    return update


def adam_update(
    grad: Tensor,
    buf1: Tensor,
    buf2: Tensor,
    step: int,
    betas: Tuple[float, float],
    eps: float,
) -> Tensor:
    """
    Perform an Adam update step.

    This function implements the standard Adam optimizer update:
    1. Update exponential moving averages of gradients and squared gradients
    2. Apply bias correction
    3. Compute the update using the Adam formula

    Args:
        grad: Gradient tensor
        buf1: First moment buffer (exponential moving average of gradients)
        buf2: Second moment buffer (exponential moving average of squared gradients)
        step: Current optimization step
        betas: Tuple of (beta1, beta2) for momentum and variance
        eps: Small constant for numerical stability

    Returns:
        Adam update tensor
    """
    # Update exponential moving averages
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])

    # Apply bias correction
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)

    # Compute Adam update
    return buf1c / (buf2c.sqrt() + eps)


class Muon(Optimizer):
    """
    Muon optimizer - MomentUm Orthogonalized by Newton-schulz.

    Muon combines standard SGD-momentum with an orthogonalization post-processing step.
    Each 2D parameter's update is replaced with the nearest orthogonal matrix using
    efficient Newton-Schulz iteration.

    This optimizer is designed for hidden weight layers in neural networks. For best results:
    - Use Muon for hidden matrix weights (2D+ parameters)
    - Use AdamW for embeddings, output layers, biases, and gains
    - For convolutional weights, Muon automatically handles the reshaping
    - 1D parameters (like biases) are handled gracefully without orthogonalization

    Args:
        params: Iterable of parameters or tensors to optimize (e.g., model.parameters())
        lr: Learning rate in units of spectral norm per update (default: 0.02)
        weight_decay: AdamW-style weight decay coefficient (default: 0)
        momentum: Momentum coefficient (default: 0.95)
        ns_steps: Number of Newton-Schulz iteration steps (default: 5)

    Example:
        >>> model = MyModel()
        >>> optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)
        >>> for epoch in range(num_epochs):
        ...     optimizer.zero_grad()
        ...     loss = model(data)
        ...     loss.backward()
        ...     optimizer.step()

    References:
        - Paper: https://kellerjordan.github.io/posts/muon/
        - Implementation: Based on work by Keller Jordan and contributors
    """

    def __init__(
        self,
        params: Union[
            Iterable[Union[Parameter, Tensor]], List[Union[Parameter, Tensor]]
        ],
        lr: float = 0.02,
        weight_decay: float = 0,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ):
        if not isinstance(lr, (int, float)) or lr < 0:
            raise ValueError(f"Learning rate must be a non-negative number, got {lr}")
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            raise ValueError(
                f"Weight decay must be a non-negative number, got {weight_decay}"
            )
        if not isinstance(momentum, (int, float)) or not 0 <= momentum < 1:
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError(f"ns_steps must be a positive integer, got {ns_steps}")

        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps
        )

        # Convert to list if it's an iterable
        if not isinstance(params, list):
            params = list(params)

        # Validate parameters
        if len(params) < 1:
            raise ValueError(
                "params must be a non-empty iterable of Parameters or Tensors"
            )
        if not all(isinstance(p, (Parameter, torch.Tensor)) for p in params):
            raise ValueError(
                "All params must be torch.nn.Parameter or torch.Tensor instances"
            )

        # Sort parameters by size for efficient distributed processing
        params = sorted(params, key=lambda x: x.size(), reverse=True)

        super().__init__(params, defaults)

        # Log initialization
        logger.info(f"Initialized Muon optimizer with {len(params)} parameters")
        logger.info(
            f"Learning rate: {lr}, Weight decay: {weight_decay}, Momentum: {momentum}"
        )

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            The loss value if closure is provided, None otherwise

        Raises:
            RuntimeError: If distributed training is not properly initialized
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]

            # Handle distributed training
            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()

                # Pad parameters for even distribution
                params_pad = params + [torch.empty_like(params[-1])] * (
                    world_size - len(params) % world_size
                )

                for base_i in range(0, len(params), world_size):
                    if base_i + rank < len(params):
                        p = params[base_i + rank]

                        # Handle missing gradients
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)

                        # Initialize state if needed
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)

                        # Perform Muon update
                        update = muon_update(
                            p.grad,
                            state["momentum_buffer"],
                            beta=group["momentum"],
                            ns_steps=group["ns_steps"],
                        )

                        # Apply weight decay and update
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])

                    # Synchronize across processes
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                    )
            else:
                # Single device processing
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                    )

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


class SingleDeviceMuon(Optimizer):
    """
    Single-device variant of the Muon optimizer.

    This optimizer is designed for non-distributed training scenarios and provides
    the same orthogonalized momentum updates as the distributed Muon optimizer.
    It handles 1D parameters (like biases) gracefully without orthogonalization.

    Args:
        params: Iterable of parameters or tensors to optimize (e.g., model.parameters())
        lr: Learning rate in units of spectral norm per update (default: 0.02)
        weight_decay: AdamW-style weight decay coefficient (default: 0)
        momentum: Momentum coefficient (default: 0.95)
        ns_steps: Number of Newton-Schulz iteration steps (default: 5)

    Example:
        >>> model = MyModel()
        >>> optimizer = SingleDeviceMuon(model.parameters(), lr=0.02)
        >>> for epoch in range(num_epochs):
        ...     optimizer.zero_grad()
        ...     loss = model(data)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        params: Union[
            Iterable[Union[Parameter, Tensor]], List[Union[Parameter, Tensor]]
        ],
        lr: float = 0.02,
        weight_decay: float = 0,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ):
        if not isinstance(lr, (int, float)) or lr < 0:
            raise ValueError(f"Learning rate must be a non-negative number, got {lr}")
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            raise ValueError(
                f"Weight decay must be a non-negative number, got {weight_decay}"
            )
        if not isinstance(momentum, (int, float)) or not 0 <= momentum < 1:
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError(f"ns_steps must be a positive integer, got {ns_steps}")

        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps
        )
        super().__init__(params, defaults)

        logger.info(
            f"Initialized SingleDeviceMuon optimizer with {len(list(params))} parameters"
        )

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            The loss value if closure is provided, None otherwise
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_steps"],
                )

                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


class MuonWithAuxAdam(Optimizer):
    """
    Hybrid optimizer combining Muon and AdamW for different parameter groups.

    This optimizer allows using Muon for matrix parameters (2D+ matrices) and AdamW for
    scalar parameters, embeddings, and output layers. This provides the benefits
    of orthogonalized updates where appropriate while maintaining compatibility
    with all parameter types.

    Args:
        param_groups: List of parameter group dictionaries. Each group must contain:
            - params: List of parameters
            - use_muon: Boolean indicating whether to use Muon (True) or AdamW (False)
            - Additional parameters specific to the chosen optimizer

    Example:
        >>> # Separate parameters by type
        >>> hidden_matrix_params = [p for n, p in model.blocks.named_parameters()
        ...                         if p.ndim >= 2 and "embed" not in n]
        >>> embed_params = [p for n, p in model.named_parameters() if "embed" in n]
        >>> scalar_params = [p for p in model.parameters() if p.ndim < 2]
        >>> head_params = [model.lm_head.weight]
        >>>
        >>> # Create parameter groups
        >>> adam_groups = [
        ...     dict(params=head_params, lr=0.22, use_muon=False),
        ...     dict(params=embed_params, lr=0.6, use_muon=False),
        ...     dict(params=scalar_params, lr=0.04, use_muon=False)
        ... ]
        >>> muon_group = dict(params=hidden_matrix_params, lr=0.05, use_muon=True)
        >>> param_groups = [*adam_groups, muon_group]
        >>>
        >>> optimizer = MuonWithAuxAdam(param_groups)

    Note:
        This optimizer requires distributed training to be initialized when used
        in a distributed setting.
    """

    def __init__(self, param_groups: List[Dict[str, Any]]):
        # Validate parameter groups
        for i, group in enumerate(param_groups):
            if "use_muon" not in group:
                raise ValueError(f"Parameter group {i} must contain 'use_muon' key")
            if "params" not in group:
                raise ValueError(f"Parameter group {i} must contain 'params' key")

            if group["use_muon"]:
                # Muon-specific validation and defaults
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["ns_steps"] = group.get("ns_steps", 5)

                expected_keys = {
                    "params",
                    "lr",
                    "momentum",
                    "weight_decay",
                    "ns_steps",
                    "use_muon",
                }
                if set(group.keys()) != expected_keys:
                    raise ValueError(
                        f"Muon parameter group {i} must contain exactly: {expected_keys}"
                    )
            else:
                # AdamW-specific validation and defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)

                expected_keys = {
                    "params",
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "use_muon",
                }
                if set(group.keys()) != expected_keys:
                    raise ValueError(
                        f"AdamW parameter group {i} must contain exactly: {expected_keys}"
                    )

        super().__init__(param_groups, dict())

        # Log initialization
        muon_params = sum(len(g["params"]) for g in param_groups if g["use_muon"])
        adam_params = sum(len(g["params"]) for g in param_groups if not g["use_muon"])
        logger.info(
            f"Initialized MuonWithAuxAdam with {muon_params} Muon parameters and {adam_params} AdamW parameters"
        )

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            The loss value if closure is provided, None otherwise
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # Muon optimization path
                params = group["params"]

                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()

                    params_pad = params + [torch.empty_like(params[-1])] * (
                        world_size - len(params) % world_size
                    )

                    for base_i in range(0, len(params), world_size):
                        if base_i + rank < len(params):
                            p = params[base_i + rank]

                            if p.grad is None:
                                p.grad = torch.zeros_like(p)

                            state = self.state[p]
                            if len(state) == 0:
                                state["momentum_buffer"] = torch.zeros_like(p)

                            update = muon_update(
                                p.grad,
                                state["momentum_buffer"],
                                beta=group["momentum"],
                                ns_steps=group["ns_steps"],
                            )

                            p.mul_(1 - group["lr"] * group["weight_decay"])
                            p.add_(update.reshape(p.shape), alpha=-group["lr"])

                        dist.all_gather(
                            params_pad[base_i : base_i + world_size],
                            params_pad[base_i + rank],
                        )
                else:
                    # Single device Muon processing
                    for p in params:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)

                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)

                        update = muon_update(
                            p.grad,
                            state["momentum_buffer"],
                            beta=group["momentum"],
                            ns_steps=group["ns_steps"],
                        )

                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                # AdamW optimization path
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(Optimizer):
    """
    Single-device variant of MuonWithAuxAdam.

    This optimizer provides the same hybrid Muon/AdamW functionality as MuonWithAuxAdam
    but is designed for non-distributed training scenarios.

    Args:
        param_groups: List of parameter group dictionaries. Each group must contain:
            - params: List of parameters
            - use_muon: Boolean indicating whether to use Muon (True) or AdamW (False)
            - Additional parameters specific to the chosen optimizer

    Example:
        >>> # Create parameter groups for single device
        >>> hidden_params = [p for p in model.parameters() if p.ndim >= 2]
        >>> scalar_params = [p for p in model.parameters() if p.ndim < 2]
        >>>
        >>> param_groups = [
        ...     dict(params=hidden_params, lr=0.02, use_muon=True),
        ...     dict(params=scalar_params, lr=3e-4, use_muon=False)
        ... ]
        >>> optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    """

    def __init__(self, param_groups: List[Dict[str, Any]]):
        # Validate parameter groups
        for i, group in enumerate(param_groups):
            if "use_muon" not in group:
                raise ValueError(f"Parameter group {i} must contain 'use_muon' key")
            if "params" not in group:
                raise ValueError(f"Parameter group {i} must contain 'params' key")

            if group["use_muon"]:
                # Muon-specific validation and defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["ns_steps"] = group.get("ns_steps", 5)

                expected_keys = {
                    "params",
                    "lr",
                    "momentum",
                    "weight_decay",
                    "ns_steps",
                    "use_muon",
                }
                if set(group.keys()) != expected_keys:
                    raise ValueError(
                        f"Muon parameter group {i} must contain exactly: {expected_keys}"
                    )
            else:
                # AdamW-specific validation and defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)

                expected_keys = {
                    "params",
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "use_muon",
                }
                if set(group.keys()) != expected_keys:
                    raise ValueError(
                        f"AdamW parameter group {i} must contain exactly: {expected_keys}"
                    )

        super().__init__(param_groups, dict())

        # Log initialization
        muon_params = sum(len(g["params"]) for g in param_groups if g["use_muon"])
        adam_params = sum(len(g["params"]) for g in param_groups if not g["use_muon"])
        logger.info(
            f"Initialized SingleDeviceMuonWithAuxAdam with {muon_params} Muon parameters and {adam_params} AdamW parameters"
        )

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            The loss value if closure is provided, None otherwise
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # Muon optimization path
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                    )

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                # AdamW optimization path
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


# Convenience function for creating parameter groups
def create_muon_param_groups(
    model: torch.nn.Module,
    muon_lr: float = 0.02,
    adam_lr: float = 3e-4,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    adam_betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0,
    eps: float = 1e-10,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups for MuonWithAuxAdam from a PyTorch model.

    This function automatically separates parameters into Muon-compatible (2D+ matrices)
    and AdamW-compatible (1D parameters, scalars, embeddings, output layers) groups.

    Args:
        model: PyTorch model to extract parameters from
        muon_lr: Learning rate for Muon parameters
        adam_lr: Learning rate for AdamW parameters
        muon_momentum: Momentum for Muon parameters
        muon_ns_steps: Number of Newton-Schulz steps for Muon parameters
        adam_betas: Beta parameters for AdamW
        weight_decay: Weight decay coefficient
        eps: Epsilon for AdamW numerical stability

    Returns:
        List of parameter group dictionaries ready for MuonWithAuxAdam

    Example:
        >>> model = MyModel()
        >>> param_groups = create_muon_param_groups(model)
        >>> optimizer = MuonWithAuxAdam(param_groups)
    """
    # Separate parameters by type
    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if param.ndim >= 2 and "embed" not in name and "head" not in name:
            muon_params.append(param)
        else:
            adam_params.append(param)

    # Create parameter groups
    param_groups = [
        # Muon group
        {
            "params": muon_params,
            "lr": muon_lr,
            "momentum": muon_momentum,
            "weight_decay": weight_decay,
            "ns_steps": muon_ns_steps,
            "use_muon": True,
        },
        # AdamW group
        {
            "params": adam_params,
            "lr": adam_lr,
            "betas": adam_betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_muon": False,
        },
    ]

    logger.info(
        f"Created parameter groups: {len(muon_params)} Muon parameters, {len(adam_params)} AdamW parameters"
    )

    return param_groups


# Version information
__version__ = "1.0.0"
__author__ = "Based on implementation by Keller Jordan and contributors"
__license__ = "MIT"

# Export main classes
__all__ = [
    "Muon",
    "SingleDeviceMuon",
    "MuonWithAuxAdam",
    "SingleDeviceMuonWithAuxAdam",
    "create_muon_param_groups",
    "zeropower_via_newtonschulz5",
    "muon_update",
    "adam_update",
]
