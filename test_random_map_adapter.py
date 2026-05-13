import unittest

import torch

import train_gpt_mps_experiment as mps


class RandomMapAdapterTests(unittest.TestCase):
    def test_zero_init_adapter_is_transparent(self):
        block = mps.Block(
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
        adapted = mps.Block(
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
            random_map_adapter=True,
            random_map_dim=4,
            random_map_seed=7,
        )
        adapted.load_state_dict(block.state_dict(), strict=False)
        x = torch.randn(2, 5, 8)
        x0 = torch.randn(2, 5, 8)

        self.assertTrue(torch.allclose(adapted(x, x0), block(x, x0), atol=1e-6, rtol=1e-6))

    def test_nonzero_adapter_changes_output(self):
        block = mps.Block(
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
            random_map_adapter=True,
            random_map_dim=4,
            random_map_seed=7,
        )
        with torch.no_grad():
            block.random_map_out.fill_(0.1)
            block.random_map_scale.fill_(1.0)
        x = torch.randn(2, 5, 8)
        x0 = torch.randn(2, 5, 8)

        self.assertGreater(block(x, x0).abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
