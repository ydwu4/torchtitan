# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import io
import os
import subprocess
import sys
import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    run_traced_module,
    trace_module,
)


def get_loss(logits, labels):
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
    )


class TrainStepModule(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, *args):
        *fwd_args, labels = args
        logits = self.model(*fwd_args)
        loss = self.loss_fn(logits, labels)
        # Must look up params in forward (not __init__) so that
        # _reparametrize_module's swapped parameters are captured during tracing.
        params = [p for _, p in self.model.named_parameters(remove_duplicate=False)]
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)


def _get_params_and_buffers(mod):
    return {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }


def create_model(config_cls, model_config, device="cuda", dtype=torch.float32):
    model = config_cls(model_config)
    model.to(device=device, dtype=dtype)
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device(device))
    return model


class SimpleMLP(nn.Module):
    def __init__(self, dim=64, hidden=128, vocab_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(self.embed(x))))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTraceModule(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32
    BATCH_SIZE = 2
    SEQ_LEN = 128
    NUM_STEPS = 5
    LR = 1e-3

    def setUp(self):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def _make_mlp(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(
            0, 256, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        labels = torch.randint(
            0, 256, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        return model, tokens, labels, get_loss

    def test_mlp_forward(self):
        model, tokens, labels, loss_fn = self._make_mlp()
        traced_result = trace_module(model, (tokens,))
        out_eager = model(tokens)
        pab = _get_params_and_buffers(model)
        wrapped = run_traced_module(traced_result, pab, (tokens,))
        self.assertTrue(torch.equal(out_eager, wrapped[0]))

    def test_mlp_train_step(self):
        model_ref, tokens, labels, loss_fn = self._make_mlp()
        model_copy = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_copy.load_state_dict(model_ref.state_dict())

        train_step = TrainStepModule(model_ref, loss_fn)
        traced_result = trace_module(train_step, (tokens, labels))

        logits_ref = model_ref(tokens)
        loss_ref = loss_fn(logits_ref, labels)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]

        train_step_copy = TrainStepModule(model_copy, loss_fn)
        pab = _get_params_and_buffers(train_step_copy)
        wrapped = run_traced_module(traced_result, pab, (tokens, labels))
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]

        self.assertTrue(torch.equal(loss_ref, loss_tr))
        for gr, gt in zip(grads_ref, grads_tr, strict=True):
            self.assertTrue(torch.equal(gr, gt))

    def test_mlp_multistep_bitwise(self):
        model_ref, tokens, labels, loss_fn = self._make_mlp()
        model_copy = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_copy.load_state_dict(model_ref.state_dict())

        train_step_ref = TrainStepModule(model_ref, loss_fn)
        train_step_copy = TrainStepModule(model_copy, loss_fn)
        traced_result = trace_module(train_step_ref, (tokens, labels))

        opt_ref = torch.optim.Adam(model_ref.parameters(), lr=self.LR)
        opt_copy = torch.optim.Adam(model_copy.parameters(), lr=self.LR)

        for step in range(1, self.NUM_STEPS + 1):
            logits_ref = model_ref(tokens)
            loss_ref = loss_fn(logits_ref, labels)
            loss_ref.backward()
            grads_ref = [p.grad.clone() for p in model_ref.parameters()]
            opt_ref.step()
            opt_ref.zero_grad()

            pab = _get_params_and_buffers(train_step_copy)
            wrapped = run_traced_module(traced_result, pab, (tokens, labels))
            loss_tr = wrapped[0]
            grads_tr = wrapped[1:]
            for p, g in zip(model_copy.parameters(), grads_tr, strict=True):
                p.grad = g
            opt_copy.step()
            opt_copy.zero_grad()

            self.assertTrue(
                torch.equal(loss_ref, loss_tr),
                f"Step {step}: loss mismatch",
            )
            self.assertTrue(
                all(
                    torch.equal(gr, gt)
                    for gr, gt in zip(grads_ref, grads_tr, strict=True)
                ),
                f"Step {step}: grad mismatch",
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTraceDTensor(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32

    def setUp(self):
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:12357",
                world_size=1,
                rank=0,
            )
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        import torch.distributed as dist

        torch.use_deterministic_algorithms(False)
        if dist.is_initialized():
            dist.destroy_process_group()

    def _distribute_params(self, model, mesh):
        from torch.distributed._tensor import distribute_tensor, Replicate

        for name, param in list(model.named_parameters()):
            dt = distribute_tensor(param, mesh, [Replicate()])
            param_parts = name.split(".")
            mod = model
            for part in param_parts[:-1]:
                mod = getattr(mod, part)
            setattr(mod, param_parts[-1], nn.Parameter(dt))

    def test_dtensor_forward(self):
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(self.DEVICE, (1,))

        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        self._distribute_params(model, mesh)

        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        tokens_dt = DTensor.from_local(tokens, mesh, [Replicate()])

        traced_result = trace_module(model, (tokens_dt,))
        has_subclass = any(
            layout.meta is not None for layout in traced_result.input_subclass_layouts
        )
        self.assertTrue(has_subclass)

        out_eager = model(tokens_dt)
        pab = _get_params_and_buffers(model)
        wrapped = run_traced_module(traced_result, pab, (tokens_dt,))
        self.assertTrue(
            torch.equal(out_eager.full_tensor(), wrapped[0].full_tensor())
        )

    def test_dtensor_train_step(self):
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(self.DEVICE, (1,))

        model_ref = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_copy = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_copy.load_state_dict(model_ref.state_dict())

        self._distribute_params(model_ref, mesh)
        self._distribute_params(model_copy, mesh)

        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        tokens_dt = DTensor.from_local(tokens, mesh, [Replicate()])
        labels_dt = DTensor.from_local(labels, mesh, [Replicate()])

        train_step = TrainStepModule(model_ref, get_loss)
        traced_result = trace_module(train_step, (tokens_dt, labels_dt))

        logits_ref = model_ref(tokens_dt)
        loss_ref = get_loss(logits_ref, labels_dt)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]

        train_step_copy = TrainStepModule(model_copy, get_loss)
        pab = _get_params_and_buffers(train_step_copy)
        wrapped = run_traced_module(traced_result, pab, (tokens_dt, labels_dt))
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]

        self.assertTrue(
            torch.equal(loss_ref.full_tensor(), loss_tr.full_tensor())
        )
        for gr, gt in zip(grads_ref, grads_tr, strict=True):
            self.assertTrue(torch.equal(gr.full_tensor(), gt.full_tensor()))


@contextlib.contextmanager
def _use_raw_flex_attn():
    from torch.nn.attention.flex_attention import flex_attention as raw_flex_attention

    from torchtitan.models.common.attention import FlexAttentionWrapper

    original = FlexAttentionWrapper._compiled_flex_attn
    FlexAttentionWrapper._compiled_flex_attn = staticmethod(raw_flex_attention)
    try:
        yield
    finally:
        FlexAttentionWrapper._compiled_flex_attn = original


def _run_bitwise_test(
    model_ref,
    model_copy,
    fwd_args,
    labels,
    check_collective_ops=False,
    num_steps=5,
    lr=1e-3,
):
    train_step_ref = TrainStepModule(model_ref, get_loss)

    with _use_raw_flex_attn():
        traced_result = trace_module(train_step_ref, (*fwd_args, labels))

    if check_collective_ops:
        ag = sum(
            1
            for n in traced_result.gm.graph.nodes
            if "all_gather_into_tensor" in str(n.target)
        )
        rs = sum(
            1
            for n in traced_result.gm.graph.nodes
            if "reduce_scatter_tensor" in str(n.target)
        )
        assert (
            ag > 0 and rs > 0
        ), f"Expected collective ops in FSDP graph (ag={ag}, rs={rs})"

    opt_ref = torch.optim.Adam(model_ref.parameters(), lr=lr)
    opt_copy = torch.optim.Adam(model_copy.parameters(), lr=lr)

    for step in range(1, num_steps + 1):
        with _use_raw_flex_attn():
            logits_ref = model_ref(*fwd_args)
        loss_ref = get_loss(logits_ref, labels)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]
        opt_ref.step()
        opt_ref.zero_grad()

        train_step_copy = TrainStepModule(model_copy, get_loss)
        pab = _get_params_and_buffers(train_step_copy)
        wrapped = run_traced_module(traced_result, pab, (*fwd_args, labels))
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]
        for p, g in zip(model_copy.parameters(), grads_tr, strict=True):
            p.grad = g
        opt_copy.step()
        opt_copy.zero_grad()

        assert torch.equal(loss_ref, loss_tr), f"Step {step}: loss mismatch"
        assert all(
            torch.equal(gr, gt)
            for gr, gt in zip(grads_ref, grads_tr, strict=True)
        ), f"Step {step}: grad mismatch"

    return True


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTraceModels(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32
    BATCH_SIZE = 2
    SEQ_LEN = 128
    NUM_STEPS = 5
    LR = 1e-3

    def setUp(self):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def _run_model_test(self, config_cls, model_config, use_attn_masks=False):
        vocab_size = model_config.vocab_size
        model_ref = create_model(config_cls, model_config, self.DEVICE, self.DTYPE)
        model_copy = create_model(config_cls, model_config, self.DEVICE, self.DTYPE)
        model_copy.load_state_dict(model_ref.state_dict())
        tokens = torch.randint(
            0, vocab_size, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        labels = torch.randint(
            0, vocab_size, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )

        if use_attn_masks:
            from torchtitan.models.common.attention import (
                create_attention_mask,
                get_causal_mask_mod,
            )

            attn_masks = create_attention_mask(
                get_causal_mask_mod(), 1, None, self.SEQ_LEN, self.SEQ_LEN
            )
            return _run_bitwise_test(
                model_ref,
                model_copy,
                (tokens, attn_masks),
                labels,
                num_steps=self.NUM_STEPS,
                lr=self.LR,
            )

        return _run_bitwise_test(
            model_ref,
            model_copy,
            (tokens,),
            labels,
            num_steps=self.NUM_STEPS,
            lr=self.LR,
        )

    def test_llama3(self):
        from torchtitan.models.llama3 import llama3_configs, Llama3Model

        self.assertTrue(self._run_model_test(Llama3Model, llama3_configs["debugmodel"]))

    def test_qwen3(self):
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.model import Qwen3Model

        self.assertTrue(self._run_model_test(Qwen3Model, qwen3_configs["debugmodel"]))

    def test_qwen3_moe(self):
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.model import Qwen3Model

        self.assertTrue(
            self._run_model_test(Qwen3Model, qwen3_configs["debugmodel_moe"])
        )

    def test_deepseek_v3(self):
        from torchtitan.models.deepseek_v3 import deepseekv3_configs
        from torchtitan.models.deepseek_v3.model import DeepSeekV3Model

        self.assertTrue(
            self._run_model_test(DeepSeekV3Model, deepseekv3_configs["debugmodel"])
        )

    def test_llama4(self):
        from torchtitan.models.llama4 import llama4_configs
        from torchtitan.models.llama4.model import Llama4Model

        self.assertTrue(
            self._run_model_test(
                Llama4Model, llama4_configs["debugmodel"], use_attn_masks=True
            )
        )

    def test_gpt_oss(self):
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.models.common.attention import (
            create_attention_mask,
            get_causal_mask_mod,
            get_sliding_window_mask_mod,
        )
        from torchtitan.models.gpt_oss import gptoss_configs
        from torchtitan.models.gpt_oss.model import GptOssModel

        config = gptoss_configs["debugmodel"]
        vocab_size = config.vocab_size
        model_ref = create_model(GptOssModel, config, self.DEVICE, self.DTYPE)
        model_copy = create_model(GptOssModel, config, self.DEVICE, self.DTYPE)
        model_copy.load_state_dict(model_ref.state_dict())
        tokens = torch.randint(
            0, vocab_size, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        labels = torch.randint(
            0, vocab_size, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        causal = get_causal_mask_mod()
        sw_size = config.layer.attention.sliding_window_size
        basic_mask = create_attention_mask(causal, 1, None, self.SEQ_LEN, self.SEQ_LEN)
        sliding_window_mask = create_attention_mask(
            and_masks(causal, get_sliding_window_mask_mod(sw_size)),
            1,
            None,
            self.SEQ_LEN,
            self.SEQ_LEN,
        )
        attn_masks = {
            "basic_mask": basic_mask,
            "sliding_window_mask": sliding_window_mask,
        }
        self.assertTrue(
            _run_bitwise_test(
                model_ref,
                model_copy,
                (tokens, attn_masks),
                labels,
                num_steps=self.NUM_STEPS,
                lr=self.LR,
            )
        )


def _run_fsdp_test(
    name,
    config_cls,
    model_config,
    use_attn_masks=False,
    device="cuda",
    dtype=torch.float32,
    batch_size=2,
    seq_len=128,
):
    import torch.distributed as dist

    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel

    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    try:
        world_size = dist.get_world_size()
        model_ref = create_model(config_cls, model_config, device, dtype)
        model_copy = create_model(config_cls, model_config, device, dtype)
        model_copy.load_state_dict(model_ref.state_dict())

        parallel_dims = ParallelDims(
            dp_shard=world_size,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size,
        )
        parallel_dims.build_mesh()
        fsdp_mesh = parallel_dims.get_mesh("fsdp")
        data_parallel(model_ref, device_mesh=fsdp_mesh, mode="fully_shard")
        data_parallel(model_copy, device_mesh=fsdp_mesh, mode="fully_shard")

        vocab_size = model_config.vocab_size
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        attn_masks = None
        if use_attn_masks:
            from torchtitan.models.common.attention import (
                create_attention_mask,
                get_causal_mask_mod,
            )

            attn_masks = create_attention_mask(
                get_causal_mask_mod(), 1, None, seq_len, seq_len
            )

        fwd_args = (tokens,) if attn_masks is None else (tokens, attn_masks)
        ctx = (
            contextlib.redirect_stdout(io.StringIO())
            if rank != 0
            else contextlib.nullcontext()
        )
        with ctx:
            return _run_bitwise_test(
                model_ref, model_copy, fwd_args, labels, check_collective_ops=True
            )
    finally:
        dist.destroy_process_group()


FSDP_REGISTRY = {
    "llama3_fsdp": lambda: _run_fsdp_test(
        "llama3 debugmodel (fsdp)",
        __import__(
            "torchtitan.models.llama3", fromlist=["Llama3Model", "llama3_configs"]
        ).Llama3Model,
        __import__(
            "torchtitan.models.llama3", fromlist=["llama3_configs"]
        ).llama3_configs["debugmodel"],
    ),
    "qwen3_fsdp": lambda: _run_fsdp_test(
        "qwen3 debugmodel (fsdp)",
        __import__("torchtitan.models.qwen3.model", fromlist=["Qwen3Model"]).Qwen3Model,
        __import__("torchtitan.models.qwen3", fromlist=["qwen3_configs"]).qwen3_configs[
            "debugmodel"
        ],
    ),
    "deepseek_v3_fsdp": lambda: _run_fsdp_test(
        "deepseek_v3 debugmodel (fsdp)",
        __import__(
            "torchtitan.models.deepseek_v3.model", fromlist=["DeepSeekV3Model"]
        ).DeepSeekV3Model,
        __import__(
            "torchtitan.models.deepseek_v3", fromlist=["deepseekv3_configs"]
        ).deepseekv3_configs["debugmodel"],
    ),
    "llama4_fsdp": lambda: _run_fsdp_test(
        "llama4 debugmodel (fsdp)",
        __import__(
            "torchtitan.models.llama4.model", fromlist=["Llama4Model"]
        ).Llama4Model,
        __import__(
            "torchtitan.models.llama4", fromlist=["llama4_configs"]
        ).llama4_configs["debugmodel"],
        use_attn_masks=True,
    ),
}


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else list(FSDP_REGISTRY.keys())

    if len(models) == 1 and models[0] in FSDP_REGISTRY:
        model_name = models[0]
        if "LOCAL_RANK" not in os.environ:
            torchrun = os.path.join(os.path.dirname(sys.executable), "torchrun")
            result = subprocess.run(
                [torchrun, "--nproc_per_node=8", __file__, model_name],
                capture_output=False,
                timeout=300,
            )
            return result.returncode
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        passed = FSDP_REGISTRY[model_name]()
        return 0 if passed else 1

    results = {}
    for model_name in models:
        print(f"\n--- Running {model_name} in subprocess ---")
        torchrun = os.path.join(os.path.dirname(sys.executable), "torchrun")
        cmd = [torchrun, "--nproc_per_node=8", __file__, model_name]
        result = subprocess.run(cmd, capture_output=False, timeout=300)
        results[model_name] = result.returncode == 0

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for model_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {model_name}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in FSDP_REGISTRY:
        sys.exit(main())
    unittest.main()
