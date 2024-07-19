# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.nn.functional as F

import triton
import triton.language as tl
import math
from torch.utils.checkpoint import checkpoint

torch.set_float32_matmul_precision("high")
device = torch.device("cuda:0")


def cosim(x, y):
    return (
        (x.reshape(-1).double() * y.reshape(-1).double()).sum()
        / x.reshape(-1).double().norm()
        / y.reshape(-1).double().norm()
    ).float()


def baseline_torch(x, y, A, ignore_index=5, z_regularization=0.0, logit_scale=1.0):
    V = A.shape[0]
    logits = F.linear(x, A).view(-1, V).div_(logit_scale).float()
    loss = F.cross_entropy(logits, y.view(-1), ignore_index=ignore_index)
    z_reg = logits.logsumexp(dim=-1)[y != ignore_index].pow(2).mean()
    loss += z_regularization * z_reg
    return loss, z_reg.detach()


def reference_torch(x, y, A, ignore_index=5, z_regularization=0.0, logit_scale=1.0):
    V = A.shape[0]
    logits = F.linear(x, A).view(-1, V).float() / logit_scale
    loss = F.cross_entropy(logits, y.view(-1), ignore_index=ignore_index)
    z_reg = logits.logsumexp(dim=-1)[y != ignore_index].pow(2).mean()
    loss += z_regularization * z_reg
    log_probs = torch.log_softmax(logits, dim=-1)[y != ignore_index]
    logit_ent = (-log_probs.exp() * log_probs).sum(dim=-1).mean()
    return loss, z_reg.detach(), logits.max().detach(), logit_ent.detach(), logits.norm(dim=-1).mean().detach()


def _inner_function(x_block, y_block, A, num_blocks, ignore_index=-1, z_regularization=0.0, logit_scale=1.0):
    V = A.shape[0]
    logits = F.linear(x_block, A).view(-1, V).float() / logit_scale
    loss = F.cross_entropy(logits, y_block.view(-1), ignore_index=ignore_index) / num_blocks
    z_reg = logits.logsumexp(dim=-1)[y_block != ignore_index].pow(2).mean() / num_blocks
    loss += z_regularization * z_reg
    probs = torch.softmax(logits, dim=-1)[y_block != ignore_index]
    logit_ent = (-probs * logits.log_softmax(dim=-1)[y_block != ignore_index]).sum(dim=-1).mean() / num_blocks
    return (
        loss,
        z_reg.detach(),
        logits.max().detach(),
        logit_ent.detach(),
        logits.norm(dim=-1).mean().detach() / num_blocks,
    )


def reference_torch_checkpoint(x, y, A, default_chunk_size=512, ignore_index=-1, z_regularization=0.0, logit_scale=1.0):
    loss = 0.0
    _, H = A.shape
    N = x.view(-1, H).shape[0]
    chunk_size = min(default_chunk_size, N)
    if chunk_size % N != 0:
        chunk_size = math.gcd(N, default_chunk_size)
    x_blocks = x.view(-1, H).split(chunk_size)
    y_blocks = y.view(-1).split(chunk_size)

    loss, z_reg, lmax, lent, lnorm = 0.0, 0.0, torch.tensor(-10e5, dtype=x.dtype), 0.0, 0.0
    for x_block, y_block in zip(x_blocks, y_blocks):
        loss_val, z_reg_val, logit_max, logit_ent, logit_norm = checkpoint(
            _inner_function,
            x_block,
            y_block,
            A,
            num_blocks=len(y_blocks),
            ignore_index=ignore_index,
            z_regularization=z_regularization,
            logit_scale=logit_scale,
            use_reentrant=False,
        )  # type: ignore
        loss += loss_val
        z_reg += z_reg_val
        lmax = torch.maximum(lmax, logit_max)
        lent += logit_ent
        lnorm += logit_norm
    return loss, z_reg, lmax, lent, lnorm


def simple_bench(fn, reference_loss, reference_x_grad, reference_A_grad, num_trials=5):
    loss_triton = fn()[0]
    with torch.autocast("cuda", enabled=False):
        loss_triton.backward()  # warmup
    torch.cuda.synchronize()
    x.grad, At.grad, A.grad = None, None, None
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # start_event.record()
    # loss_triton = fn(x, y, At)
    # end_event.record()
    # torch.cuda.synchronize()
    # estimate_ms_fwd = start_event.elapsed_time(end_event)
    # print(f"fwd : {estimate_ms_fwd}ms")
    # print(f"fwd error: {torch.dist(loss_triton, reference_loss).item()}")

    # loss_triton.backward(retain_graph=True)  # warmup
    # x.grad, At.grad = None, None
    # torch.cuda.synchronize()
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    estimate_ms_bwd = 0
    for _ in range(num_trials):
        x.grad, At.grad, A.grad = None, None, None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # type: ignore
        loss_triton = fn()[0]
        with torch.autocast("cuda", enabled=False):
            loss_triton.backward()
        end_event.record()  # type: ignore
        torch.cuda.synchronize()
        estimate_ms_bwd += start_event.elapsed_time(end_event) / num_trials
    print(f"fwd-bwd : {estimate_ms_bwd}ms")
    print(f"fwd error: {torch.dist(loss_triton, reference_loss).item()}")
    if At.grad is not None:
        A_error = torch.dist(reference_A_grad.T, At.grad).item()
    else:
        A_error = torch.dist(reference_A_grad, A.grad).item()  # type: ignore
    print(f"bwd error: {torch.dist(reference_x_grad, x.grad).item()}, {A_error}")  # type: ignore


def early_config_prune(configs, named_args, **kwargs):
    dtype = named_args["x_ptr"].dtype

    # 1. make sure blocks are small enough
    N, H, V = named_args["x_ptr"].shape[0], named_args["x_ptr"].shape[1], named_args["A_t_ptr"].shape[1]
    pruned_configs = []
    for config in configs:
        accept = True
        accept &= config.kwargs.get("SPLIT_V", 1) * config.kwargs["V_BLOCK_SIZE"] <= V
        accept &= config.kwargs.get("SPLIT_N", 1) * config.kwargs["N_BLOCK_SIZE"] <= N
        accept &= config.kwargs["H_BLOCK_SIZE"] <= H
        if accept:
            pruned_configs.append(config)
    configs = pruned_configs

    # Some dtypes do not allow atomic_add
    if dtype not in [torch.float16, torch.float32]:
        configs = [
            config
            for config in configs
            if (config.kwargs.get("SPLIT_N", 1) == 1) or (config.kwargs.get("SPLIT_N", 1) == 1)
        ]
    # print(len(configs))
    if len(configs) == 0:
        raise ValueError("Provided shape outside the range of valid configs.")
    return configs


fwd_configs = [
    triton.Config(
        {"V_BLOCK_SIZE": 512, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64, "GROUP_SIZE": 1}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64, "GROUP_SIZE": 32}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64, "GROUP_SIZE": 16}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64, "GROUP_SIZE": 1}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 512, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64, "GROUP_SIZE": 16}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64, "GROUP_SIZE": 1}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64, "GROUP_SIZE": 16}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64, "GROUP_SIZE": 1}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 512, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64, "GROUP_SIZE": 32}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 64, "GROUP_SIZE": 1}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 64, "GROUP_SIZE": 64}, num_warps=8, num_stages=2
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 64, "GROUP_SIZE": 64}, num_warps=16, num_stages=2
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 128, "GROUP_SIZE": 1}, num_warps=8, num_stages=2
    ),
]
for num_stages in [2, 3]:
    for warps in [4, 8]:
        for v_block in [256, 512]:
            for n_block in [128]:
                for h_block in [128]:
                    for group in [64, 32, 16]:
                        fwd_configs.append(
                            triton.Config(
                                {
                                    "V_BLOCK_SIZE": v_block,
                                    "N_BLOCK_SIZE": n_block,
                                    "H_BLOCK_SIZE": h_block,
                                    "GROUP_SIZE": group,
                                },
                                num_warps=warps,
                                num_stages=num_stages,
                            )
                        )


@triton.autotune(
    configs=fwd_configs,
    key=["V", "N", "H", "monitoring"],
    # prune_configs_by={
    #     "early_config_prune": early_config_prune,
    #     # "perf_model": lambda: 1,
    #     # "top_k": 10,
    # },
    # warmup=100,
    # rep=500,
)
@triton.jit
def linear_xent_fwd_prep_bwd_kernel_matmul_t(
    x_ptr,
    y_ptr,
    A_t_ptr,
    z_nv_ptr,
    losses_ptr,
    lse_ptr,
    m_ptr,
    logit_norm_ptr,
    stride_x_N,
    stride_x_H,
    stride_A_H,
    stride_A_V,
    stride_z_N,
    stride_z_V,
    stride_lse_N,
    stride_lse_B,
    stride_loss_Nb,
    stride_loss_B,
    stride_norm_N,
    stride_norm_V,
    reduction_ptr,
    monitoring: tl.constexpr,
    ignore_index: tl.constexpr,  # let's bake this in
    logit_scale: tl.constexpr,
    idx_N_group,
    N_group: tl.constexpr,
    V: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    V_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
    H_BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    idx_N = tl.program_id(axis=0)
    idx_V_group = tl.program_id(axis=1)
    num_idx_N, num_idx_V_group = tl.num_programs(0), tl.num_programs(1)
    idx_N, idx_V_group = tl.swizzle2d(idx_N, idx_V_group, num_idx_N, num_idx_V_group, GROUP_SIZE)  # type:ignore

    tl.static_print(V_BLOCK_SIZE, N_BLOCK_SIZE, H_BLOCK_SIZE, GROUP_SIZE)

    R = tl.load(reduction_ptr, eviction_policy="evict_last")
    V_GROUP_SIZE: tl.constexpr = V_BLOCK_SIZE
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, H),
        strides=(stride_x_N, stride_x_H),
        offsets=(idx_N_group * N_group + idx_N * N_BLOCK_SIZE, 0),
        block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE),
        order=(1, 0),
    )
    A_block_ptr = tl.make_block_ptr(
        base=A_t_ptr,
        shape=(H, V),
        strides=(stride_A_H, stride_A_V),
        offsets=(0, idx_V_group * V_GROUP_SIZE),
        block_shape=(H_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0),  # (0, 1) apparently not faster :<
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_nv_ptr,
        shape=(N_group, V),
        strides=(stride_z_N, stride_z_V),
        offsets=(idx_N * N_BLOCK_SIZE, idx_V_group * V_GROUP_SIZE),
        block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0),
    )

    z_j_to_k = tl.zeros((N_BLOCK_SIZE, V_BLOCK_SIZE), dtype=tl.float32)
    for _ in range(H // H_BLOCK_SIZE):
        x_chunk = tl.load(x_block_ptr)  # Nc x H
        A_v = tl.load(A_block_ptr)  # Vc x H

        z_j_to_k = tl.dot(x_chunk, A_v, z_j_to_k)  # (Nc x H) @ (H x Vc)

        x_block_ptr = tl.advance(x_block_ptr, [0, H_BLOCK_SIZE])
        A_block_ptr = tl.advance(A_block_ptr, [H_BLOCK_SIZE, 0])

    z_j_to_k = z_j_to_k / logit_scale
    if monitoring:
        logit_pow2 = tl.sum(z_j_to_k * z_j_to_k, axis=1)
        norm_val_ptr = logit_norm_ptr + idx_V_group * stride_norm_V + idx_N * stride_norm_N + tl.arange(0, N_BLOCK_SIZE)
        tl.store(norm_val_ptr, logit_pow2 / N)
    m = tl.max(z_j_to_k, 1)
    s = tl.sum(tl.exp((z_j_to_k - m[:, None])), axis=1)

    N_range = idx_N_group * N_group + idx_N * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)
    V_range = idx_V_group * V_GROUP_SIZE + tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + N_range)

    mask = y[:, None] == tl.where(V_range != ignore_index, V_range, -1)[None, :]  # Nc x Vc
    loss = -tl.sum(tl.where(mask, z_j_to_k, 0.0)) / R

    # save z for later
    tl.store(z_block_ptr, z_j_to_k.to(z_nv_ptr.type.element_ty))  # can move +log(1/N) here

    zero_lse_constant: tl.constexpr = tl.log(1 / tl.cdiv(V, V_BLOCK_SIZE))  # type: ignore
    lse = tl.where(y != ignore_index, m + tl.log(s), zero_lse_constant)
    lse_row_ptr = tl.make_block_ptr(
        base=lse_ptr,
        shape=(N_group, V // 128),  # fixed to largest number of possible V blocks
        strides=(stride_lse_N, stride_lse_B),
        offsets=(idx_N * N_BLOCK_SIZE, idx_V_group),
        block_shape=(N_BLOCK_SIZE, 1),
        order=(1, 0),
    )
    tl.store(lse_row_ptr, lse[:, None])

    loss_val_ptr = losses_ptr + idx_N * stride_loss_Nb + idx_V_group * stride_loss_B
    # loss += tl.sum(lse) / N # defered until all blocks are done
    tl.store(loss_val_ptr, tl.load(loss_val_ptr) + loss)

    if monitoring:
        m_val_ptr = m_ptr + idx_N * stride_loss_Nb + idx_V_group * stride_loss_B
        tl.store(m_val_ptr, tl.maximum(tl.load(m_val_ptr), tl.max(m, 0)))


@triton.jit()
def linear_xent_bwd_kernel_matmul_t_epilogue_dx(
    z_nv_ptr,
    y_ptr,
    A_t_ptr,
    x_grad_ptr,
    lse_ptr,
    stride_x_N,
    stride_x_H,
    stride_A_H,
    stride_A_V,
    stride_z_N,
    stride_z_V,
    reduction_ptr,
    logit_scale: tl.constexpr,
    z_regularization: tl.constexpr,
    fp32_grad_accumulators: tl.constexpr,
    ignore_index: tl.constexpr,
    idx_N_group,
    N_group: tl.constexpr,
    V: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    V_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
    H_BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT_N: tl.constexpr,
    SPLIT_V: tl.constexpr,
):
    idx_N = tl.program_id(axis=0) // SPLIT_V
    idx_H = tl.program_id(axis=1)
    idx_V_tile = tl.program_id(axis=0) % SPLIT_V

    num_idx_N = tl.num_programs(0) - (triton.cdiv(V, V_BLOCK_SIZE) * SPLIT_N)  # type: ignore
    num_idx_H = tl.num_programs(1)
    idx_N, idx_H = tl.swizzle2d(idx_N, idx_H, num_idx_N // SPLIT_V, num_idx_H, GROUP_SIZE)  # type:ignore

    V_split_offset = idx_V_tile * tl.cdiv(V, SPLIT_V)

    A_t_block_ptr = tl.make_block_ptr(
        base=A_t_ptr,
        shape=(H, V),
        strides=(stride_A_H, stride_A_V),
        offsets=(idx_H * H_BLOCK_SIZE, V_split_offset),
        block_shape=(H_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(0, 1),
    )

    z_block_ptr = tl.make_block_ptr(
        base=z_nv_ptr,
        shape=(N_group, V),
        strides=(stride_z_N, stride_z_V),
        offsets=(idx_N * N_BLOCK_SIZE, V_split_offset),
        block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0),
    )
    N_range = idx_N_group * N_group + idx_N * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)
    v_range = V_split_offset + tl.arange(0, V_BLOCK_SIZE)
    R = tl.load(reduction_ptr, eviction_policy="evict_last")

    y = tl.load(y_ptr + N_range, eviction_policy="evict_last")
    lse = tl.load(lse_ptr + idx_N * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE), eviction_policy="evict_last")

    acc_dtype = tl.float32 if fp32_grad_accumulators else x_grad_ptr.type.element_ty
    x_grad_acc = tl.zeros((N_BLOCK_SIZE, H_BLOCK_SIZE), acc_dtype)
    for _ in range(0, tl.cdiv(V, V_BLOCK_SIZE * SPLIT_V)):
        mask = y[:, None] == v_range[None, :]
        A_v = tl.load(A_t_block_ptr, eviction_policy="evict_first")  # Hc x Vc
        z_j_to_k = tl.load(z_block_ptr)
        softmax_z = (z_j_to_k - lse[:, None]).exp()

        if z_regularization > 0:
            softmax_z += 2.0 * z_regularization * lse[:, None] * softmax_z
        z_grad = softmax_z - tl.where(mask, 1.0, 0.0)  # 1/N, 0 if log(1/N) moved
        valid_z_grad = tl.where((y == ignore_index)[:, None], 0.0, z_grad).to(A_v.type.element_ty)

        # xgrad
        x_grad_acc = tl.dot(valid_z_grad, A_v.trans(), x_grad_acc, out_dtype=acc_dtype)

        A_t_block_ptr = tl.advance(A_t_block_ptr, [0, V_BLOCK_SIZE])
        z_block_ptr = tl.advance(z_block_ptr, [0, V_BLOCK_SIZE])
        v_range += V_BLOCK_SIZE

    if SPLIT_V == 1:
        x_grad_block_ptr = tl.make_block_ptr(
            base=x_grad_ptr,
            shape=(N, H),
            strides=(stride_x_N, stride_x_H),
            offsets=(idx_N_group * N_group + idx_N * N_BLOCK_SIZE, idx_H * H_BLOCK_SIZE),
            block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE),
            order=(1, 0),
        )
        tl.store(x_grad_block_ptr, (x_grad_acc / R / logit_scale).to(x_grad_ptr.type.element_ty))
        # not divided here if 1/N moved
    else:
        row_n = idx_N_group * N_group + idx_N * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)
        row_h = idx_H * H_BLOCK_SIZE + tl.arange(0, H_BLOCK_SIZE)
        x_grad_simple_ptr = x_grad_ptr + row_n[:, None] * stride_x_N + row_h[None, :] * stride_x_H
        tl.atomic_add(x_grad_simple_ptr, (x_grad_acc / R / logit_scale).to(x_grad_ptr.type.element_ty))


@triton.jit()
def linear_xent_bwd_kernel_matmul_t_epilogue_dA(
    z_nv_ptr,
    y_ptr,
    x_ptr,
    A_grad_ptr,
    lse_ptr,
    entropy_ptr,
    stride_x_N,
    stride_x_H,
    stride_A_H,
    stride_A_V,
    stride_z_N,
    stride_z_V,
    stride_ent_H,
    stride_ent_V,
    reduction_ptr,
    monitoring: tl.constexpr,
    logit_scale: tl.constexpr,
    z_regularization: tl.constexpr,
    fp32_grad_accumulators: tl.constexpr,
    ignore_index: tl.constexpr,
    idx_N_group,
    N_group: tl.constexpr,
    V: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    V_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
    H_BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SPLIT_N: tl.constexpr,
    SPLIT_V: tl.constexpr,
):
    idx_V = (tl.program_id(axis=0) - N_group // N_BLOCK_SIZE * SPLIT_V) // SPLIT_N
    idx_H = tl.program_id(axis=1)
    idx_N_tile = (tl.program_id(axis=0) - N_group // N_BLOCK_SIZE * SPLIT_V) % SPLIT_N

    num_idx_V, num_idx_H = tl.num_programs(0) - (N_group // N_BLOCK_SIZE * SPLIT_V), tl.num_programs(1)
    idx_V, idx_H = tl.swizzle2d(idx_V, idx_H, num_idx_V // SPLIT_N, num_idx_H, GROUP_SIZE)  # type:ignore

    N_split_offset = idx_N_tile * tl.cdiv(N_group, SPLIT_N)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, H),
        strides=(stride_x_N, stride_x_H),
        offsets=(idx_N_group * N_group + N_split_offset, idx_H * H_BLOCK_SIZE),
        block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE),
        order=(1, 0),
    )

    z_block_ptr = tl.make_block_ptr(
        base=z_nv_ptr,
        shape=(N_group, V),
        strides=(stride_z_N, stride_z_V),
        offsets=(N_split_offset, idx_V * V_BLOCK_SIZE),
        block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0),
    )

    N_range = N_split_offset + tl.arange(0, N_BLOCK_SIZE)
    V_range = idx_V * V_BLOCK_SIZE + tl.arange(0, V_BLOCK_SIZE)
    R = tl.load(reduction_ptr, eviction_policy="evict_last")
    logit_entropy = 0.0

    acc_dtype = tl.float32 if fp32_grad_accumulators else A_grad_ptr.type.element_ty
    A_grad_acc = tl.zeros((H_BLOCK_SIZE, V_BLOCK_SIZE), acc_dtype)
    for _ in range(0, tl.cdiv(N_group, N_BLOCK_SIZE * SPLIT_N)):
        y = tl.load(y_ptr + idx_N_group * N_group + N_range, eviction_policy="evict_last")
        lse = tl.load(lse_ptr + N_range, eviction_policy="evict_last")
        mask = y[:, None] == V_range[None, :]

        x_chunk = tl.load(x_block_ptr, eviction_policy="evict_first")
        z_j_to_k = tl.load(z_block_ptr)
        logprobs = z_j_to_k - lse[:, None]
        softmax_z = logprobs.exp()
        if monitoring:
            logit_entropy += tl.sum(tl.where(y == ignore_index, 0.0, tl.sum(-softmax_z * logprobs, axis=1)))
        if z_regularization > 0:
            softmax_z += 2.0 * z_regularization * lse[:, None] * softmax_z
        z_grad = softmax_z - tl.where(mask, 1.0, 0.0)
        valid_z_grad = tl.where((y == ignore_index)[:, None], 0.0, z_grad).to(x_ptr.type.element_ty)

        A_grad_acc = tl.dot(x_chunk.trans(), valid_z_grad, A_grad_acc, out_dtype=acc_dtype)

        x_block_ptr = tl.advance(x_block_ptr, [N_BLOCK_SIZE, 0])
        z_block_ptr = tl.advance(z_block_ptr, [N_BLOCK_SIZE, 0])
        N_range += N_BLOCK_SIZE

    entropy_val_ptr = entropy_ptr + idx_H * stride_ent_H + idx_V * stride_ent_V
    if SPLIT_N == 1:
        A_grad_T_block_ptr = tl.make_block_ptr(
            base=A_grad_ptr,
            shape=(H, V),
            strides=(stride_A_H, stride_A_V),
            offsets=(idx_H * H_BLOCK_SIZE, idx_V * V_BLOCK_SIZE),
            block_shape=(H_BLOCK_SIZE, V_BLOCK_SIZE),
            order=(0, 1),
        )
        if idx_N_group > 0:
            tl.store(
                A_grad_T_block_ptr,
                tl.load(A_grad_T_block_ptr) + (A_grad_acc / R / logit_scale).to(A_grad_ptr.type.element_ty),
            )
            tl.store(entropy_val_ptr, tl.load(entropy_val_ptr) + logit_entropy / R)
        else:
            tl.store(A_grad_T_block_ptr, (A_grad_acc / R / logit_scale).to(A_grad_ptr.type.element_ty))
            if monitoring:
                tl.store(entropy_val_ptr, logit_entropy / R)
    else:
        row_h = idx_H * H_BLOCK_SIZE + tl.arange(0, H_BLOCK_SIZE)
        row_v = idx_V * V_BLOCK_SIZE + tl.arange(0, V_BLOCK_SIZE)
        A_grad_T_simple_ptr = A_grad_ptr + row_h[:, None] * stride_A_H + row_v[None, :] * stride_A_V
        tl.atomic_add(A_grad_T_simple_ptr, (A_grad_acc / R / logit_scale).to(A_grad_ptr.type.element_ty))
        if monitoring:
            tl.atomic_add(entropy_val_ptr, logit_entropy / R)


bwd_configs = [
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 128, "GROUP_SIZE": 16, "SPLIT_N": 1, "SPLIT_V": 1},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 256, "GROUP_SIZE": 32, "SPLIT_N": 1, "SPLIT_V": 4},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 256, "GROUP_SIZE": 16, "SPLIT_N": 1, "SPLIT_V": 1},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 128, "GROUP_SIZE": 32, "SPLIT_N": 1, "SPLIT_V": 1},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 256, "GROUP_SIZE": 32, "SPLIT_N": 1, "SPLIT_V": 1},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 128, "GROUP_SIZE": 16, "SPLIT_N": 1, "SPLIT_V": 1},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 256, "GROUP_SIZE": 16, "SPLIT_N": 1, "SPLIT_V": 2},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 256, "GROUP_SIZE": 16, "SPLIT_N": 1, "SPLIT_V": 4},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 256, "GROUP_SIZE": 64, "SPLIT_N": 1, "SPLIT_V": 4},
        num_warps=16,
        num_stages=2,
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 256, "GROUP_SIZE": 64, "SPLIT_N": 1, "SPLIT_V": 8},
        num_warps=16,
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 256, "GROUP_SIZE": 64, "SPLIT_N": 1, "SPLIT_V": 8},
        num_warps=8,
    ),
    triton.Config(
        {"V_BLOCK_SIZE": 128, "N_BLOCK_SIZE": 128, "H_BLOCK_SIZE": 256, "GROUP_SIZE": 64, "SPLIT_N": 1, "SPLIT_V": 4},
        num_warps=8,
        num_stages=2,
    ),
]

for num_stages in [2, 3]:
    for warps in [8, 16, 32]:
        for v_block in [128, 256]:
            for n_block in [128]:
                for h_block in [128, 256]:
                    for group in [32, 64]:
                        for split_n in [1]:
                            for split_v in [4, 8, 32]:
                                bwd_configs.append(
                                    triton.Config(
                                        {
                                            "V_BLOCK_SIZE": v_block,
                                            "N_BLOCK_SIZE": n_block,
                                            "H_BLOCK_SIZE": h_block,
                                            "GROUP_SIZE": group,
                                            "SPLIT_N": split_n,
                                            "SPLIT_V": split_v,
                                        },
                                        num_warps=warps,
                                        num_stages=num_stages,
                                    )
                                )


@triton.autotune(
    configs=bwd_configs,
    key=["V", "N", "H", "monitoring"],
    # prune_configs_by={
    #     "early_config_prune": early_config_prune,
    #     # "perf_model": lambda: 1,
    #     # "top_k": 10,
    # },
    # warmup=100,
    # rep=500,
)
@triton.jit()
def linear_xent_bwd_dispatcher(
    logits_ptr,
    y_ptr,
    x_ptr,
    A_t_ptr,
    x_grad,
    At_grad,
    lse_global,
    logit_entropy_local,
    stride_x_N,
    stride_x_H,
    stride_A_H,
    stride_A_V,
    stride_z_N,
    stride_z_V,
    stride_ent_H,
    stride_ent_V,
    reduction_ptr,
    monitoring: tl.constexpr,
    logit_scale: tl.constexpr,
    z_regularization: tl.constexpr,
    fp32_grad_accumulators: tl.constexpr,
    ignore_index: tl.constexpr,
    idx_N_group,
    N_group: tl.constexpr,
    V: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    V_BLOCK_SIZE: tl.constexpr = 128,  # type: ignore
    N_BLOCK_SIZE: tl.constexpr = 128,  # type: ignore
    H_BLOCK_SIZE: tl.constexpr = 128,  # type: ignore
    GROUP_SIZE: tl.constexpr = 32,  # type: ignore
    SPLIT_N: tl.constexpr = 2,  # type: ignore
    SPLIT_V: tl.constexpr = 2,  # type: ignore
):
    tl.static_print(V_BLOCK_SIZE, N_BLOCK_SIZE, H_BLOCK_SIZE, GROUP_SIZE, SPLIT_N, SPLIT_V)

    idx_NV = tl.program_id(axis=0)
    if idx_NV < (N_group // N_BLOCK_SIZE * SPLIT_V):
        linear_xent_bwd_kernel_matmul_t_epilogue_dx(
            logits_ptr,
            y_ptr,
            A_t_ptr,
            x_grad,
            lse_global,
            stride_x_N,
            stride_x_H,
            stride_A_H,
            stride_A_V,
            stride_z_N,
            stride_z_V,
            reduction_ptr,
            logit_scale,
            z_regularization,
            fp32_grad_accumulators,
            ignore_index,
            idx_N_group,
            N_group,
            V,
            N,
            H,
            V_BLOCK_SIZE,
            N_BLOCK_SIZE,
            H_BLOCK_SIZE,
            GROUP_SIZE,
            SPLIT_N,
            SPLIT_V,
        )
    else:
        linear_xent_bwd_kernel_matmul_t_epilogue_dA(
            logits_ptr,
            y_ptr,
            x_ptr,
            At_grad,
            lse_global,
            logit_entropy_local,
            stride_x_N,
            stride_x_H,
            stride_A_H,
            stride_A_V,
            stride_z_N,
            stride_z_V,
            stride_ent_H,
            stride_ent_V,
            reduction_ptr,
            monitoring,
            logit_scale,
            z_regularization,
            fp32_grad_accumulators,
            ignore_index,
            idx_N_group,
            N_group,
            V,
            N,
            H,
            V_BLOCK_SIZE,
            N_BLOCK_SIZE,
            H_BLOCK_SIZE,
            GROUP_SIZE,
            SPLIT_N,
            SPLIT_V,
        )


# @triton.autotune(
#     configs=[
#         triton.Config({"N_BLOCK_SIZE": 2}, num_warps=4, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 4}, num_warps=4, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 8}, num_warps=2, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 8}, num_warps=4, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 8}, num_warps=8, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 8}, num_warps=2, num_stages=4),
#         triton.Config({"N_BLOCK_SIZE": 8}, num_warps=4, num_stages=4),
#         triton.Config({"N_BLOCK_SIZE": 8}, num_warps=8, num_stages=4),
#         triton.Config({"N_BLOCK_SIZE": 8}, num_warps=16, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 16}, num_warps=2, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 16}, num_warps=4, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 16}, num_warps=8, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 16}, num_warps=16, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 32}, num_warps=2, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 32}, num_warps=4, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 32}, num_warps=8, num_stages=3),
#         triton.Config({"N_BLOCK_SIZE": 32}, num_warps=16, num_stages=5),
#         triton.Config({"N_BLOCK_SIZE": 16}, num_warps=4, num_stages=2),
#         triton.Config({"N_BLOCK_SIZE": 16}, num_warps=8, num_stages=2),
#         triton.Config({"N_BLOCK_SIZE": 16}, num_warps=16, num_stages=2),
#         triton.Config({"N_BLOCK_SIZE": 16}, num_warps=4, num_stages=1),
#         triton.Config({"N_BLOCK_SIZE": 16}, num_warps=8, num_stages=1),
#         triton.Config({"N_BLOCK_SIZE": 16}, num_warps=16, num_stages=1),
#     ],
#     key=["N_group", "V", "V_BLOCK_SIZE"],
# )
# @triton.jit()
# def logsumexp_reduction_kernel(
#     lse_local_ptr,
#     lse_global_ptr,
#     lse_sum_ptr,
#     z_reg_sum_ptr,
#     reduction_ptr,
#     z_regularization: tl.constexpr,
#     stride_lse_N,
#     stride_lse_B,
#     N_group,
#     V: tl.constexpr,
#     V_BLOCK_SIZE: tl.constexpr,
#     N_BLOCK_SIZE: tl.constexpr = 32,
# ):
#     idx_N = tl.program_id(axis=0)
#     lse_row_ptr = tl.make_block_ptr(
#         base=lse_local_ptr,
#         shape=(N_group, V // 128),  # fixed to worst case number assuming max(V_TILES)
#         strides=(stride_lse_N, stride_lse_B),
#         offsets=(idx_N * N_BLOCK_SIZE, 0),
#         block_shape=(N_BLOCK_SIZE, V // V_BLOCK_SIZE),
#         order=(1, 0),
#     )
#     lse_local = tl.load(lse_row_ptr)
#     m = tl.max(lse_local, 1)
#     lse = tl.log(tl.sum(tl.exp((lse_local - m[:, None])), axis=1)) + m

#     if z_regularization > 0:
#         z_reg = tl.sum(lse * lse) / tl.load(reduction_ptr)
#         lse_reduction = (tl.sum(lse) / tl.load(reduction_ptr) + z_regularization * z_reg)
#         tl.atomic_add(z_reg_sum_ptr, z_reg)
#     else:
#         lse_reduction = tl.sum(lse) / tl.load(reduction_ptr)
#     tl.atomic_add(lse_sum_ptr, lse_reduction)

#     tl.store(lse_global_ptr + idx_N * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE), lse)


class LinearXentImplementation(torch.autograd.Function):
    @torch._dynamo.disable()
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    @staticmethod
    def forward(
        ctx,
        x,
        y,
        At,
        ignore_index=-100,
        z_regularization: float = 0.0,
        logit_scale: float = 1.0,
        N_chunk_size: int = 4096,  # N_chunk_size x V is the maximal memory peak
        monitoring: bool = True,
    ):
        with torch.cuda.device(x.device.index):  # actually required for devices other than 0
            N, H = x.shape
            H_A, V = At.shape
            assert H_A == H
            assert y.shape == (N,)
            N_group = min(N, N_chunk_size)

            assert N % 64 == 0
            assert V % 128 == 0
            assert H % 64 == 0
            with torch.no_grad():
                At_grad = torch.zeros_like(At, dtype=torch.float32 if At.dtype == torch.bfloat16 else At.dtype)
                x_grad = torch.zeros_like(x, dtype=torch.float32 if At.dtype == torch.bfloat16 else At.dtype)

                lse_sum, z_reg_value, logit_norm = 0.0, 0.0, 0.0
                # lse_sum = torch.zeros((1,), dtype=torch.float32, device=x.device)
                # z_reg_value = torch.zeros((1,), dtype=torch.float32, device=x.device)

                lse_local = -10e5 * torch.ones(N_group, V // 128, dtype=torch.float32, device=x.device)
                logit_max_local = torch.ones(N_group, V // 128, dtype=torch.float32, device=x.device)
                losses = torch.zeros(N_group // 64, V // 128, dtype=torch.float32, device=x.device)
                # lse_global = torch.empty(N_group, dtype=torch.float32, device=x.device)

                logits = torch.empty((N_group, V), device=x.device, dtype=torch.float32)
                logit_ent_local = torch.zeros(V // 128, H // 64, dtype=torch.float32, device=x.device)
                logit_norm_local = torch.zeros(N_group, V // 128, dtype=torch.float32, device=x.device)

                reduction = (y != ignore_index).sum()
                if reduction == 0:
                    ctx.mark_non_differentiable(y)
                    ctx.save_for_backward(x_grad, At_grad.to(At.dtype))
                    return losses.sum()

                fwd_grid = lambda meta: (
                    triton.cdiv(N_group, meta["N_BLOCK_SIZE"]),
                    triton.cdiv(V, meta["V_BLOCK_SIZE"]),
                )
                bwd_grid_dx_dA = lambda meta: (
                    triton.cdiv(N_group, meta["N_BLOCK_SIZE"]) * meta["SPLIT_V"]
                    + triton.cdiv(V, meta["V_BLOCK_SIZE"]) * meta["SPLIT_N"],
                    triton.cdiv(H, meta["H_BLOCK_SIZE"]),
                )
                # bwd_prologue_grid = lambda meta: (triton.cdiv(N_group, meta["N_BLOCK_SIZE"]),)
                for idx_N_group in range(math.ceil(N / N_group)):
                    linear_xent_fwd_prep_bwd_kernel_matmul_t[fwd_grid](
                        x,
                        y,
                        At,
                        logits,
                        losses,
                        lse_local,
                        logit_max_local,
                        logit_norm_local,
                        x.stride(0),
                        x.stride(1),
                        At.stride(0),
                        At.stride(1),
                        logits.stride(0),
                        logits.stride(1),
                        lse_local.stride(0),
                        lse_local.stride(1),
                        losses.stride(0),
                        losses.stride(1),
                        logit_norm_local.stride(0),
                        logit_norm_local.stride(1),
                        reduction,
                        monitoring=monitoring,
                        logit_scale=logit_scale,
                        ignore_index=ignore_index,
                        idx_N_group=idx_N_group,
                        N_group=N_group,
                        V=V,
                        N=N,
                        H=H,
                    )
                    logit_norm += logit_norm_local.sum(dim=-1).sqrt().sum()

                    # V_BLOCK_SIZE = linear_xent_fwd_prep_bwd_kernel_matmul_t.best_config.kwargs["V_BLOCK_SIZE"]
                    # logsumexp_reduction_kernel[bwd_prologue_grid](
                    #     lse_local,
                    #     lse_global,
                    #     lse_sum,
                    #     z_reg_value,
                    #     reduction,
                    #     z_regularization,
                    #     lse_local.stride(0),
                    #     lse_local.stride(1),
                    #     N_group=N_group,
                    #     V=V,
                    #     V_BLOCK_SIZE=V_BLOCK_SIZE,  # not independent, from prev kernel
                    # )

                    # buffer_extent = V // V_BLOCK_SIZE
                    lse_global = lse_local.logsumexp(dim=1)
                    z_reg_block = lse_global.pow(2).sum()
                    if z_regularization > 0:
                        lse_sum += (lse_global.sum() + z_regularization * z_reg_block) / reduction
                    else:
                        lse_sum += lse_global.sum() / reduction
                    z_reg_value += z_reg_block / reduction

                    # if x.requires_grad or At.requires_grad:
                    linear_xent_bwd_dispatcher[bwd_grid_dx_dA](
                        logits,
                        y,
                        x,
                        At,
                        x_grad,
                        At_grad,
                        lse_global,
                        logit_ent_local,
                        x_grad.stride(0),
                        x_grad.stride(1),
                        At.stride(0),
                        At.stride(1),
                        logits.stride(0),
                        logits.stride(1),
                        logit_ent_local.stride(0),
                        logit_ent_local.stride(1),
                        reduction,
                        monitoring=monitoring,
                        logit_scale=logit_scale,
                        z_regularization=z_regularization,
                        fp32_grad_accumulators=x.dtype == torch.bfloat16,
                        ignore_index=ignore_index,
                        idx_N_group=idx_N_group,
                        N_group=N_group,
                        V=V,
                        N=N,
                        H=H,
                    )

                logit_max = logit_max_local.max()
                logit_ent = logit_ent_local.sum()
                ctx.mark_non_differentiable(z_reg_value, logit_max, logit_ent, logit_norm)
                ctx.save_for_backward(x_grad, At_grad)
                print("fwd", linear_xent_fwd_prep_bwd_kernel_matmul_t.best_config)
                print("bwd", linear_xent_bwd_dispatcher.best_config)
            return lse_sum + losses.sum()  # , z_reg_value, logit_max, logit_ent, logit_norm

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    @torch.inference_mode()
    def backward(ctx, grad_output):
        x_grad, At_grad = ctx.saved_tensors

        return x_grad * grad_output, None, At_grad * grad_output, None, None, None, None, None


# @torch.compile
def linear_cross_entropy(
    x,
    y,
    At,
    ignore_index=-100,
    z_regularization: float = 0.0,
    logit_scale: float = 1.0,
    N_chunk_size: int = 4096,
    monitoring: bool = False,
):
    return LinearXentImplementation.apply(x, y, At, ignore_index, z_regularization, logit_scale, N_chunk_size, monitoring)  # type: ignore


class LinearCrossEntropyLoss(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
        ignore_index: int = -100,
        logit_scale: float = 1.0,
        z_regularization: float = 0.0,
        N_chunk_size: int = 4096,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features), **factory_kwargs))  # transposed!

        self.logit_scale = logit_scale
        self.ignore_index = ignore_index

        assert in_features % 128 == 0
        assert out_features % 128 == 0
        assert N_chunk_size % 128 == 0
        assert logit_scale > 0
        assert z_regularization > 0

        self.monitoring = False
        self.latest_metrics = {}

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(1 / self.in_features)
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    @torch.compile
    def forward(self, x, y):
        loss, z_reg, logit_max, logit_ent, logit_norm = LinearXentImplementation.apply(
            x,
            y,
            self.weight,
            self.ignore_index,
            self.z_regularization,
            self.logit_scale,
            self.N_chunk_size,
            self.monitoring,
        )  # type: ignore
        if self.monitoring:
            metrics = {
                "logit_norm": logit_norm,
                "logit_max": logit_max,
                "logit_entropy": logit_ent,
                "z_value": z_reg,
            }
        self.latest_metrics = metrics  # will be picked up from monitoring caller
        return loss


if __name__ == "__main__":
    f = 1
    # V, N, H = 4096 * f, 512 * f, 256 * f
    #  V, N, H = 32768 * f, 4096 * f, 2048 * f
    V, N, H = 32768 * 2, 4096 * 8, 1024
    compute_dtype = torch.float16

    z_reg_const = 0
    logit_scale = math.sqrt(H)
    ignore_index = 5

    y = torch.randint(0, V, (N,), device=device)  # vocab ** B S
    y[0:256] = 5
    A = torch.randn(V, H, requires_grad=True, device=device, dtype=compute_dtype)
    At = A.clone().detach().T.contiguous()
    At.requires_grad_()

    x = (0.1 * A[y].clone().detach() + torch.randn(N, H, device=device, dtype=compute_dtype)) * 1
    x.requires_grad_()

    A_ref = A.clone().detach()

    loss, z_reg_ref, logit_max_ref, logit_ent_ref, logit_norm_ref = reference_torch_checkpoint(
        x.float(), y, A.float(), ignore_index=ignore_index, z_regularization=z_reg_const, logit_scale=logit_scale
    )
    loss.backward()  # type: ignore

    reference_A_grad = A.grad.float().clone()  # type: ignore
    reference_x_grad = x.grad.float().clone()  # type: ignore
    reference_loss = loss.detach().float().clone()  # type: ignore

    print(reference_loss)

    simple_bench(
        # lambda: torch.compile(linear_cross_entropy, fullgraph=True, mode="max-autotune")(x, y, At),
        lambda: linear_cross_entropy(
            x, y, At, ignore_index=ignore_index, z_regularization=z_reg_const, logit_scale=logit_scale
        ),
        reference_loss,
        reference_x_grad,
        reference_A_grad,
    )

    _, z_reg, lmax, lent, lnorm = linear_cross_entropy(
        x, y, At, ignore_index=ignore_index, z_regularization=z_reg_const, logit_scale=logit_scale, monitoring=True
    )  # type: ignore
    print(torch.dist(z_reg, z_reg_ref), torch.dist(lmax, logit_max_ref), torch.dist(lent, logit_ent_ref), torch.dist(lnorm, logit_norm_ref))  # type: ignore

    print("ref norm ", logit_norm_ref)
    print("triton norm", lnorm, logit_norm_ref / lnorm)

    print("ref ent ", logit_ent_ref)
    print("triton ent", lent, logit_ent_ref / lent)
    simple_bench(
        lambda: torch.compile(baseline_torch)(
            x, y, A, ignore_index=ignore_index, z_regularization=z_reg_const, logit_scale=logit_scale
        ),
        reference_loss,
        reference_x_grad,
        reference_A_grad,
    )

    x = x.float().detach().requires_grad_()
    At = At.float().detach().requires_grad_()
    # test with AMP:
    print("with autocast:")
    # this is semi-correct because now the bwd pass is also wrapped
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        linear_cross_entropy(x, y, At, ignore_index=ignore_index, z_regularization=z_reg_const, logit_scale=logit_scale)
        simple_bench(
            # lambda: torch.compile(linear_cross_entropy, fullgraph=True, mode="max-autotune")(x, y, At),
            lambda: linear_cross_entropy(
                x, y, At, ignore_index=ignore_index, z_regularization=z_reg_const, logit_scale=logit_scale
            ),
            reference_loss,
            reference_x_grad,
            reference_A_grad,
        )

        _, z_reg, lmax, lent, lnorm = linear_cross_entropy(
            x, y, At, ignore_index=ignore_index, z_regularization=z_reg_const, logit_scale=logit_scale, monitoring=True
        )  # type: ignore
        print(torch.dist(z_reg, z_reg_ref), torch.dist(lmax, logit_max_ref), torch.dist(lent, logit_ent_ref), torch.dist(lnorm, logit_norm_ref))  # type: ignore

        simple_bench(
            lambda: torch.compile(baseline_torch)(
                x, y, A, ignore_index=ignore_index, z_regularization=z_reg_const, logit_scale=logit_scale
            ),
            reference_loss,
            reference_x_grad,
            reference_A_grad,
        )
