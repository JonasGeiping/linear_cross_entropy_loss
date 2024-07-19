import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


device = torch.device("cuda:0")
torch.cuda.device_count()


def cosim(x, y):
    return (
        (x.reshape(-1).double() * y.reshape(-1).double()).sum()
        / x.reshape(-1).double().norm()
        / y.reshape(-1).double().norm()
    ).float()


@torch._dynamo.disable
def baseline_torch(x, y, A):
    V = A.shape[0]
    return F.cross_entropy(F.linear(x, A).view(-1, V).float(), y.view(-1))


# a compromise


def simple_bench(fn, reference_loss, reference_x_grad, reference_A_grad):
    loss_triton = fn().backward()  # warmup
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
    start_event.record()
    loss_triton = fn()
    loss_triton.backward()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms_bwd = start_event.elapsed_time(end_event)
    print(f"fwd-bwd : {estimate_ms_bwd}ms")
    print(f"fwd error: {torch.dist(loss_triton, reference_loss).item()}")
    if At.grad is not None:
        A_error = torch.dist(reference_A_grad.T, At.grad).item()
    else:
        A_error = torch.dist(reference_A_grad, A.grad).item()
    print(f"bwd error: {torch.dist(reference_x_grad, x.grad).item()}, {A_error}")


@triton.autotune(
    configs=[
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64}),
        triton.Config({"V_BLOCK_SIZE": 64, "N_BLOCK_SIZE": 16, "H_BLOCK_SIZE": 64}),
        triton.Config({"V_BLOCK_SIZE": 64, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64}),
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 16, "H_BLOCK_SIZE": 256}),
        triton.Config({"V_BLOCK_SIZE": 512, "N_BLOCK_SIZE": 16, "H_BLOCK_SIZE": 512}),
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64}),
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 256, "H_BLOCK_SIZE": 64}),
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 256, "H_BLOCK_SIZE": 256}),
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 16, "H_BLOCK_SIZE": 16}),
    ],
    key=["V", "N", "H"],
    reset_to_zero=["losses_ptr", "lse_ptr"],
)
@triton.jit
def linear_xent_fwd_prep_bwd_kernel_matmul_t(
    x_ptr,
    y_ptr,
    A_t_ptr,
    z_nv_ptr,
    losses_ptr,
    lse_ptr,
    stride_x_N,
    stride_x_H,
    stride_A_H,
    stride_A_V,
    stride_z_N,
    stride_z_V,
    N_offset,
    V: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    V_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
    H_BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(axis=0)
    tl.static_print(V_BLOCK_SIZE, N_BLOCK_SIZE, H_BLOCK_SIZE)
    tl.static_assert(N % N_BLOCK_SIZE == 0)
    tl.static_assert(V % V_BLOCK_SIZE == 0)
    tl.static_assert(H % H_BLOCK_SIZE == 0)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, H),
        strides=(stride_x_N, stride_x_H),
        offsets=(idx * N_BLOCK_SIZE, 0),
        block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE),
        order=(1, 0),
    )
    A_block_ptr = tl.make_block_ptr(
        base=A_t_ptr,
        shape=(H, V),
        strides=(stride_A_H, stride_A_V),
        offsets=(0, 0),
        block_shape=(H_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0),
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_nv_ptr,
        shape=(N, V),
        strides=(stride_z_N, stride_z_V),
        offsets=(idx * N_BLOCK_SIZE, 0),
        block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0),
    )

    offsets = N_offset + idx * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)
    v_range = tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + offsets)

    m = tl.zeros((N_BLOCK_SIZE,), dtype=tl.float32) - float(10e5)
    s = tl.zeros((N_BLOCK_SIZE,), dtype=tl.float32)
    loss = 0.0

    for _ in range(V // V_BLOCK_SIZE):
        z_j_to_k = tl.zeros((N_BLOCK_SIZE, V_BLOCK_SIZE), dtype=tl.float32)
        for _ in range(H // H_BLOCK_SIZE):
            x_chunk = tl.load(x_block_ptr)  # Nc x H
            A_v = tl.load(A_block_ptr)  # Vc x H

            z_j_to_k = tl.dot(x_chunk, A_v, z_j_to_k)  # (Nc x H) @ (H x Vc)

            x_block_ptr = tl.advance(x_block_ptr, [0, H_BLOCK_SIZE])
            A_block_ptr = tl.advance(A_block_ptr, [H_BLOCK_SIZE, 0])

        m_new = tl.maximum(m, tl.max(z_j_to_k, 1))

        s_update = tl.sum(tl.exp(z_j_to_k - m_new[:, None]), axis=1)
        s = s * tl.exp(m - m_new) + s_update

        mask = y[:, None] == v_range[None, :]  # Nc x Vc
        loss -= tl.sum(tl.where(mask, z_j_to_k, float(0.0))) / N

        # save z for later
        tl.store(z_block_ptr, z_j_to_k.to(z_nv_ptr.type.element_ty))

        # Reset and advance pointers for next step
        m = m_new
        x_block_ptr = tl.advance(x_block_ptr, [0, -H])
        A_block_ptr = tl.advance(A_block_ptr, [-H, V_BLOCK_SIZE])
        z_block_ptr = tl.advance(z_block_ptr, [0, V_BLOCK_SIZE])
        v_range = v_range + V_BLOCK_SIZE

    lse = m + tl.log(s)
    loss += tl.sum(lse) / N
    tl.store(losses_ptr + idx, loss)
    tl.store(lse_ptr + offsets, lse)


@triton.autotune(
    configs=[
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 64}),
        triton.Config({"V_BLOCK_SIZE": 64, "N_BLOCK_SIZE": 16}),
        triton.Config({"V_BLOCK_SIZE": 64, "N_BLOCK_SIZE": 64}),
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 16}),
        triton.Config({"V_BLOCK_SIZE": 512, "N_BLOCK_SIZE": 16}),
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 64}),
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 256}),
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 256}),
        triton.Config({"V_BLOCK_SIZE": 256, "N_BLOCK_SIZE": 16}),
    ],
    key=["V", "N"],
    reset_to_zero=["z_grad_ptr"],
)
@triton.jit
def linear_xent_mini_bwd_prologue_kernel(
    z_nv_ptr,
    z_grad_ptr,
    y_ptr,
    lse_ptr,
    stride_z_N,
    stride_z_V,
    N_offset,
    V: tl.constexpr,
    N: tl.constexpr,
    V_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    idx_N = tl.program_id(axis=0)
    idx_V = tl.program_id(axis=1)
    tl.static_print(V_BLOCK_SIZE, N_BLOCK_SIZE)
    tl.static_assert(N % N_BLOCK_SIZE == 0)
    tl.static_assert(V % V_BLOCK_SIZE == 0)
    z_block_ptr = tl.make_block_ptr(
        base=z_nv_ptr,
        shape=(N, V),
        strides=(stride_z_N, stride_z_V),
        offsets=(idx_N * N_BLOCK_SIZE, idx_V * V_BLOCK_SIZE),
        block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0),
    )
    z_grad_block_ptr = tl.make_block_ptr(
        base=z_grad_ptr,
        shape=(N, V),
        strides=(stride_z_N, stride_z_V),
        offsets=(idx_N * N_BLOCK_SIZE, idx_V * V_BLOCK_SIZE),
        block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0),
    )
    N_range = N_offset + idx_N * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)
    v_range = idx_V * V_BLOCK_SIZE + tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + N_range)
    lse = tl.load(lse_ptr + N_range)
    z_j_to_k = tl.load(z_block_ptr)

    mask = y[:, None] == v_range[None, :]
    softmax_z = (z_j_to_k - lse[:, None]).exp()
    z_grad = (softmax_z - tl.where(mask, 1.0, 0.0)) / N

    tl.store(z_grad_block_ptr, z_grad.to(tl.float16))


@triton.autotune(
    configs=[
        triton.Config({"V_BLOCK_SIZE": 16, "N_BLOCK_SIZE": 16, "H_BLOCK_SIZE": 32}),
        triton.Config({"V_BLOCK_SIZE": 16, "N_BLOCK_SIZE": 16, "H_BLOCK_SIZE": 16}),
        triton.Config({"V_BLOCK_SIZE": 16, "N_BLOCK_SIZE": 16, "H_BLOCK_SIZE": 64}),
        triton.Config({"V_BLOCK_SIZE": 32, "N_BLOCK_SIZE": 32, "H_BLOCK_SIZE": 32}),
        triton.Config({"V_BLOCK_SIZE": 32, "N_BLOCK_SIZE": 32, "H_BLOCK_SIZE": 64}),
        triton.Config({"V_BLOCK_SIZE": 32, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64}),
        triton.Config({"V_BLOCK_SIZE": 64, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 64}),
        triton.Config({"V_BLOCK_SIZE": 16, "N_BLOCK_SIZE": 16, "H_BLOCK_SIZE": 128}),
        triton.Config({"V_BLOCK_SIZE": 16, "N_BLOCK_SIZE": 16, "H_BLOCK_SIZE": 256}),
        triton.Config({"V_BLOCK_SIZE": 16, "N_BLOCK_SIZE": 16, "H_BLOCK_SIZE": 512}),
        triton.Config({"V_BLOCK_SIZE": 64, "N_BLOCK_SIZE": 64, "H_BLOCK_SIZE": 16}),
        triton.Config({"V_BLOCK_SIZE": 32, "N_BLOCK_SIZE": 32, "H_BLOCK_SIZE": 16}),
    ],
    key=["V", "N", "H"],
    reset_to_zero=["x_grad_ptr"],
)
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
    N_offset,
    V: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    V_BLOCK_SIZE: tl.constexpr = 16,
    N_BLOCK_SIZE: tl.constexpr = 16,
    H_BLOCK_SIZE: tl.constexpr = 16,
):
    idx_N = tl.program_id(axis=0)
    idx_H = tl.program_id(axis=1)
    idx_V = 0

    tl.static_print(V_BLOCK_SIZE, N_BLOCK_SIZE, H_BLOCK_SIZE)
    tl.static_assert(N % N_BLOCK_SIZE == 0)
    tl.static_assert(V % V_BLOCK_SIZE == 0)

    x_grad_block_ptr = tl.make_block_ptr(
        base=x_grad_ptr,
        shape=(N, H),
        strides=(stride_x_N, stride_x_H),
        offsets=(N_offset + idx_N * N_BLOCK_SIZE, idx_H * H_BLOCK_SIZE),
        block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE),
        order=(1, 0),
    )
    A_t_block_ptr = tl.make_block_ptr(
        base=A_t_ptr,
        shape=(H, V),
        strides=(stride_A_H, stride_A_V),
        offsets=(idx_H * H_BLOCK_SIZE, 0),
        block_shape=(H_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0),
    )

    z_block_ptr = tl.make_block_ptr(
        base=z_nv_ptr,
        shape=(N, V),
        strides=(stride_z_N, stride_z_V),
        offsets=(idx_N * N_BLOCK_SIZE, 0),
        block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0),
    )
    N_range = N_offset + idx_N * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)
    v_range = 0 + tl.arange(0, V_BLOCK_SIZE)

    y = tl.load(y_ptr + N_range)
    lse = tl.load(lse_ptr + N_range)

    x_grad_acc = tl.zeros((N_BLOCK_SIZE, H_BLOCK_SIZE), tl.float32)
    for idx_V in range(V // V_BLOCK_SIZE):
        mask = (y[:, None] == v_range[None, :])[:, :, None]  # N_BLOCK_SIZE x V_BLOCK_SIZE x 1
        A_v = tl.load(A_t_block_ptr).trans()  # Hc x Vc
        z_j_to_k = tl.load(z_block_ptr)
        softmax_z = (z_j_to_k - lse[:, None]).exp().to(tl.float16)

        # xgrad
        x_grad_acc = tl.dot(softmax_z, A_v, x_grad_acc)
        x_grad_acc -= tl.sum(tl.where(mask, A_v[None, :, :], 0.0), axis=1)

        A_t_block_ptr = tl.advance(A_t_block_ptr, [0, V_BLOCK_SIZE])
        z_block_ptr = tl.advance(z_block_ptr, [0, V_BLOCK_SIZE])
        v_range += V_BLOCK_SIZE

    tl.store(x_grad_block_ptr, (x_grad_acc / N).to(tl.float16))


class LinearCrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        y,
        At,
        ignore_index=-100,  # code ignores all negative integers right now
        N_chunk_size: int = 4096,  # N_chunk_size x V is the maximal memory peak
    ):
        N, H = x.shape
        H_A, V = At.shape
        assert H_A == H
        assert y.shape == (N,)
        # x = x.contiguous()
        # y = y.contiguous()
        if ignore_index >= 0:
            y[y == ignore_index] = -100
        At = At.contiguous()
        A_grad = torch.zeros_like(At.T)
        x_grad = torch.zeros_like(x)

        lse_global = torch.zeros(N, dtype=torch.float32, device=x.device)
        losses = torch.zeros(N // 16, dtype=torch.float32, device=x.device)

        fwd_grid = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        bwd_grid_dx = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]), triton.cdiv(H, meta["H_BLOCK_SIZE"]))
        bwd_grid_dA = lambda meta: (triton.cdiv(N, meta["V_BLOCK_SIZE"]), triton.cdiv(V, meta["H_BLOCK_SIZE"]))

        for idx, x_n_chunk in enumerate(x.split(N_chunk_size)):
            x_input = x_n_chunk.contiguous()

            z_nv = torch.empty((N_chunk_size, V), device=x.device, dtype=torch.float32)

            with torch.cuda.device(x.device.index):  # actually required
                linear_xent_fwd_prep_bwd_kernel_matmul_t[fwd_grid](
                    x,
                    y,
                    At,
                    z_nv,
                    losses,
                    lse_global,
                    x.stride(0),
                    x.stride(1),
                    At.stride(0),
                    At.stride(1),
                    z_nv.stride(0),
                    z_nv.stride(1),
                    N_offset=idx * N_chunk_size,
                    V=V,
                    N=N_chunk_size,
                    H=H,
                )
                if x.requires_grad:
                    linear_xent_bwd_kernel_matmul_t_epilogue_dx[bwd_grid_dx](
                        z_nv,
                        y,
                        At,
                        x_grad,
                        lse_global,
                        x_grad.stride(0),
                        x_grad.stride(1),
                        At.stride(0),
                        At.stride(1),
                        z_nv.stride(0),
                        z_nv.stride(1),
                        idx * N_chunk_size,
                        V,
                        N_chunk_size,
                        H,
                    )

                if At.requires_grad:
                    torch.addmm(
                        A_grad,
                        z_nv.T.half(),
                        x_input,
                        out=A_grad,
                    )

        print("fwd config:", linear_xent_fwd_prep_bwd_kernel_matmul_t.best_config)
        ctx.mark_non_differentiable(y)
        ctx.save_for_backward(x_grad, A_grad.T)

        return losses.sum()

    @staticmethod
    def backward(ctx, grad_output):
        x_grad, At_grad = ctx.saved_tensors

        return x_grad * grad_output, None, At_grad * grad_output, None


def linear_cross_entropy(x, y, At):
    return LinearCrossEntropyLoss.apply(x, y, At)


if __name__ == "__main__":
    f = 1
    V, N, H = 32768, 4096 * 2, 1024

    compute_dtype = torch.float16

    y = torch.randint(0, V, (N,), device=device)  # vocab ** B S
    A = torch.randn(V, H, requires_grad=True, device=device, dtype=compute_dtype)
    At = A.clone().detach().T.contiguous()
    At.requires_grad_()

    # x = torch.randn(B * S, H, requires_grad=True, device=device, dtype=torch.float32) # B S H
    # x = A[y].clone().detach()
    x = 0.15 * A[y].clone().detach() + torch.randn(N, H, device=device, dtype=compute_dtype)
    x.requires_grad_()

    loss = baseline_torch(x.float(), y, A.float())
    loss.backward()

    reference_A_grad = A.grad.float().clone()
    reference_x_grad = x.grad.float().clone()
    reference_loss = loss.detach().float().clone()

    z_ref = F.linear(x, A).view(-1, V).float().detach()
    m_ref = z_ref.max(dim=1)[0]
    s_ref = (z_ref - m_ref[:, None]).exp().sum(dim=1)

    print(reference_loss)

    simple_bench(lambda: linear_cross_entropy(x, y, At), reference_loss, reference_x_grad, reference_A_grad)

    simple_bench(lambda: torch.compile(baseline_torch)(x, y, A), reference_loss, reference_x_grad, reference_A_grad)
