import unittest

import torch

import train_gpt_mps_experiment as mps


class CausalConvMemoryTests(unittest.TestCase):
    def _block(self):
        return mps.Block(
            8,
            2,
            2,
            2,
            10000.0,
            1.5,
            use_iha=False,
            iha_mix_v=True,
            attn_out_gate=False,
            attn_out_gate_src="proj",
            attn_out_gate_width=4,
            sparse_attn_gate=False,
            sparse_attn_gate_scale=1.0,
            causal_conv_memory=True,
            conv_kernel=3,
            conv_dilation=2,
        )

    def test_zero_init_conv_memory_is_transparent(self):
        plain = mps.Block(
            8,
            2,
            2,
            2,
            10000.0,
            1.5,
            use_iha=False,
            iha_mix_v=True,
            attn_out_gate=False,
            attn_out_gate_src="proj",
            attn_out_gate_width=4,
            sparse_attn_gate=False,
            sparse_attn_gate_scale=1.0,
        )
        conv = self._block()
        conv.load_state_dict(plain.state_dict(), strict=False)
        x = torch.randn(2, 7, 8)
        x0 = torch.randn(2, 7, 8)

        self.assertTrue(torch.allclose(conv(x, x0), plain(x, x0), atol=1e-6, rtol=1e-6))

    def test_conv_memory_is_causal(self):
        block = self._block()
        with torch.no_grad():
            block.conv_memory_weight.zero_()
            block.conv_memory_weight[:, -1].fill_(1.0)
        x = torch.randn(1, 8, 8)
        changed_future = x.clone()
        changed_future[:, 6:] += 100.0

        y = block._causal_conv_memory(x)
        y_changed = block._causal_conv_memory(changed_future)

        self.assertTrue(torch.allclose(y[:, :6], y_changed[:, :6], atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
