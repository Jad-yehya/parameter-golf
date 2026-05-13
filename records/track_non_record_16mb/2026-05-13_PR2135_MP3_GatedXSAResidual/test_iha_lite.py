import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _install_flash_attn_stub():
    def _flash_attn_func(q, k, v, causal=True):
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        if k_t.size(1) != q_t.size(1):
            repeat = q_t.size(1) // k_t.size(1)
            k_t = k_t.repeat_interleave(repeat, dim=1)
            v_t = v_t.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
        return y.transpose(1, 2)

    flash_mod = types.ModuleType("flash_attn_interface")
    flash_mod.flash_attn_func = _flash_attn_func
    flash_mod.flash_attn_varlen_func = lambda *args, **kwargs: _flash_attn_func(
        args[0].unsqueeze(0), args[1].unsqueeze(0), args[2].unsqueeze(0), kwargs.get("causal", True)
    )[0]
    sys.modules["flash_attn_interface"] = flash_mod


def _install_triton_stub_if_needed():
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
        from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    triton_mod = types.ModuleType("triton")
    triton_mod.jit = lambda fn=None, **kwargs: fn if fn is not None else (lambda f: f)
    triton_mod.cdiv = lambda a, b: (a + b - 1) // b

    tl_mod = types.ModuleType("triton.language")
    tl_mod.constexpr = object()
    tl_mod.dtype = object
    triton_mod.language = tl_mod

    tools_mod = types.ModuleType("triton.tools")
    tensor_descriptor_mod = types.ModuleType("triton.tools.tensor_descriptor")
    tensor_descriptor_mod.TensorDescriptor = object

    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = tl_mod
    sys.modules["triton.tools"] = tools_mod
    sys.modules["triton.tools.tensor_descriptor"] = tensor_descriptor_mod


def _load_train_module():
    _install_flash_attn_stub()
    _install_triton_stub_if_needed()
    old_compile = torch.compile
    torch.compile = lambda fn=None, **kwargs: fn if fn is not None else (lambda f: f)
    module_path = Path(__file__).with_name("train_gpt.py")
    spec = importlib.util.spec_from_file_location("candidate_train_gpt", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        torch.compile = old_compile
    return module


class _ZeroLoRA(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        return x.new_zeros(*x.shape[:-1], self.out_features)


class _FakeLoRA:
    def __init__(self, dim, kv_dim):
        self.q_loras = [_ZeroLoRA(dim)]
        self.k_loras = [_ZeroLoRA(kv_dim)]
        self.v_loras = [_ZeroLoRA(kv_dim)]
        self.o_loras = [_ZeroLoRA(dim)]
        self.mlp_loras = [_ZeroLoRA(dim)]


class IHALiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_train_module()

    def _weights(self, dim=8, num_heads=2, num_kv_heads=2, mlp_mult=2):
        head_dim = dim // num_heads
        kv_dim = num_kv_heads * head_dim
        hidden_dim = mlp_mult * dim
        torch.manual_seed(123)
        return (
            torch.randn(dim, dim),
            torch.randn(kv_dim, dim),
            torch.randn(kv_dim, dim),
            torch.randn(dim, dim),
            torch.randn(hidden_dim, dim),
            torch.randn(dim, hidden_dim),
        )

    def test_iha_identity_init_matches_plain_attention(self):
        torch.manual_seed(1)
        plain = self.m.CausalSelfAttention(
            8, 2, 2, 10000.0, 1.5, 16, use_iha=False
        )
        iha = self.m.CausalSelfAttention(
            8, 2, 2, 10000.0, 1.5, 16, use_iha=True, iha_mix_v=True
        )
        iha.q_gain.data.copy_(plain.q_gain.data)
        x = torch.randn(2, 5, 8)
        q_w, k_w, v_w, out_w, *_ = self._weights()

        plain_out = plain(x, q_w, k_w, v_w, out_w)
        iha_out = iha(x, q_w, k_w, v_w, out_w)

        self.assertTrue(torch.allclose(iha_out, plain_out, atol=1e-6, rtol=1e-6))

    def test_zero_lora_ttt_block_matches_normal_block_with_iha(self):
        block = self.m.Block(
            8, 2, 2, 2, 10000.0, 1.5, 16, use_iha=True, iha_mix_v=True
        )
        block.eval()
        with torch.no_grad():
            mix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            block.attn.q_head_mix.copy_(mix)
            block.attn.k_head_mix.copy_(mix)
            block.attn.v_head_mix.copy_(mix)

        torch.manual_seed(2)
        x = torch.randn(2, 5, 8)
        x0 = torch.randn(2, 5, 8)
        q_w, k_w, v_w, out_w, up_w, down_w = self._weights()
        lora = _FakeLoRA(dim=8, kv_dim=8)

        normal = block(x, x0, q_w, k_w, v_w, out_w, up_w, down_w)
        ttt = self.m.GPT._block_with_lora(
            object(), block, x, x0, lora, 0, q_w, k_w, v_w, out_w, up_w, down_w
        )

        self.assertTrue(torch.allclose(ttt, normal, atol=1e-6, rtol=1e-6))

    def test_zero_lora_parallel_ttt_block_matches_normal_parallel_block_with_iha(self):
        block = self.m.Block(
            8, 2, 2, 2, 10000.0, 1.5, 16, use_iha=True, iha_mix_v=True
        )
        block.eval()
        with torch.no_grad():
            mix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            block.attn.q_head_mix.copy_(mix)
            block.attn.k_head_mix.copy_(mix)
            block.attn.v_head_mix.copy_(mix)

        fake_gpt = types.SimpleNamespace(
            blocks=[block],
            parallel_resid_lambdas=torch.tensor([[1.1, 0.9]]),
            parallel_post_lambdas=torch.tensor([[[1.0, 0.25], [0.1, 0.95]]]),
        )
        torch.manual_seed(3)
        lane0 = torch.randn(2, 5, 8)
        lane1 = torch.randn(2, 5, 8)
        x0 = torch.randn(2, 5, 8)
        q_w, k_w, v_w, out_w, up_w, down_w = self._weights()
        lora = _FakeLoRA(dim=8, kv_dim=8)

        normal = self.m.GPT._parallel_block(
            fake_gpt, 0, lane0, lane1, x0, q_w, k_w, v_w, out_w, up_w, down_w
        )
        ttt = self.m.GPT._parallel_block_with_lora(
            fake_gpt, 0, lane0, lane1, x0, lora, 0, q_w, k_w, v_w, out_w, up_w, down_w
        )

        self.assertTrue(torch.allclose(ttt[0], normal[0], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(ttt[1], normal[1], atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
