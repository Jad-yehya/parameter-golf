import unittest

import torch

import train_gpt_mps_experiment as mps


class SparseGateTests(unittest.TestCase):
    def test_sparse_gate_is_mutually_exclusive_with_attention_output_gate(self):
        with self.assertRaises(ValueError):
            mps.CausalSelfAttention(
                8,
                2,
                2,
                10000.0,
                1.5,
                attn_out_gate=True,
                sparse_attn_gate=True,
            )

    def test_zero_init_sparse_gate_halves_attention_output(self):
        torch.manual_seed(123)
        plain = mps.CausalSelfAttention(8, 2, 2, 10000.0, 1.5)
        gated = mps.CausalSelfAttention(
            8, 2, 2, 10000.0, 1.5, sparse_attn_gate=True, attn_out_gate_width=4
        )
        gated.c_q.load_state_dict(plain.c_q.state_dict())
        gated.c_k.load_state_dict(plain.c_k.state_dict())
        gated.c_v.load_state_dict(plain.c_v.state_dict())
        gated.q_gain.data.copy_(plain.q_gain.data)
        with torch.no_grad():
            plain.proj.weight.copy_(torch.eye(8))
            gated.proj.weight.copy_(torch.eye(8))

        x = torch.randn(2, 5, 8)

        self.assertTrue(torch.allclose(gated(x), 0.5 * plain(x), atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
