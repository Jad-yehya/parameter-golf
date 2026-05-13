import unittest

import torch

import train_text_diffusion_mps as td


class FakeSentencePiece:
    def __init__(self):
        self._pieces = {
            0: "<unk>",
            1: "a",
            2: "\u2581cat",
            3: "\u2581\u00e9",
            4: "<0x41>",
        }

    def vocab_size(self):
        return len(self._pieces)

    def is_control(self, token_id):
        return False

    def is_unknown(self, token_id):
        return token_id == 0

    def is_unused(self, token_id):
        return False

    def is_byte(self, token_id):
        return token_id == 4

    def id_to_piece(self, token_id):
        return self._pieces[token_id]


class TextDiffusionHarnessTests(unittest.TestCase):
    def test_log_linear_noise_matches_alpha_identity(self):
        t = torch.tensor([0.0, 0.5, 1.0])
        sigma, alpha = td.log_linear_noise(t, eps=0.1)
        self.assertTrue(torch.allclose(alpha, torch.tensor([1.0, 0.55, 0.1])))
        self.assertTrue(torch.allclose(torch.exp(-sigma), alpha, atol=1e-6))

    def test_sentencepiece_byte_luts_count_utf8_and_leading_space(self):
        base_bytes, leading, boundary = td.build_sentencepiece_luts(
            FakeSentencePiece(), vocab_size=6, device=torch.device("cpu")
        )
        self.assertEqual(int(base_bytes[1]), 1)
        self.assertEqual(int(base_bytes[2]), 3)
        self.assertEqual(int(base_bytes[3]), 2)
        self.assertEqual(int(base_bytes[4]), 1)
        self.assertFalse(bool(leading[1]))
        self.assertTrue(bool(leading[2]))
        self.assertTrue(bool(boundary[0]))
        self.assertFalse(bool(boundary[2]))

    def test_make_corrupted_batch_supports_contiguous_span_masking(self):
        x0 = torch.arange(16).reshape(2, 8)
        t = torch.full((2,), 0.5)
        torch.manual_seed(3)
        xt, mask = td.make_corrupted_batch(x0, t, mask_id=99, eps=0.1, pattern="span", span_len=3)
        self.assertEqual(xt.shape, x0.shape)
        self.assertEqual(mask.shape, x0.shape)
        self.assertTrue(torch.equal(xt[mask], torch.full_like(xt[mask], 99)))
        for row in mask:
            idx = row.nonzero(as_tuple=False).flatten()
            if idx.numel() > 1:
                gaps = idx[1:] - idx[:-1]
                self.assertLessEqual(int(gaps.max()), 3)

    def test_tiny_model_forward_and_mdlm_loss_are_finite(self):
        cfg = td.ModelConfig(
            vocab_size=16,
            mask_id=16,
            padded_vocab=24,
            seq_len=8,
            num_layers=1,
            model_dim=32,
            num_heads=4,
            mlp_mult=2.0,
            self_condition=True,
        )
        model = td.DiffusionLM(cfg)
        x0 = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
        sigma = torch.full((2,), 0.4)
        logits = model.forward_logits(x0, sigma)
        self.assertEqual(logits.shape, (2, cfg.seq_len, cfg.total_vocab))
        loss = td.mdlm_loss(model, x0, mask_pattern="independent")
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_tied_output_head_reuses_token_embedding_parameters(self):
        untied_cfg = td.ModelConfig(
            vocab_size=16,
            mask_id=16,
            padded_vocab=24,
            seq_len=8,
            num_layers=1,
            model_dim=32,
            num_heads=4,
            mlp_mult=2.0,
            tie_output_head=False,
        )
        tied_cfg = td.ModelConfig(
            vocab_size=16,
            mask_id=16,
            padded_vocab=24,
            seq_len=8,
            num_layers=1,
            model_dim=32,
            num_heads=4,
            mlp_mult=2.0,
            tie_output_head=True,
        )
        untied = td.DiffusionLM(untied_cfg)
        tied = td.DiffusionLM(tied_cfg)
        untied_params = sum(p.numel() for p in untied.parameters())
        tied_params = sum(p.numel() for p in tied.parameters())
        self.assertEqual(untied_params - tied_params, untied_cfg.padded_vocab * untied_cfg.model_dim)
        self.assertAlmostEqual(tied.output_head_scale, tied_cfg.model_dim**-0.5)
        self.assertIs(tied.head, None)
        self.assertIsNot(tied.tok_emb.weight, None)
        self.assertNotIn("head.weight", dict(tied.named_parameters()))

    def test_tied_output_head_forward_shape_and_excludes_mask_logit(self):
        cfg = td.ModelConfig(
            vocab_size=16,
            mask_id=16,
            padded_vocab=24,
            seq_len=8,
            num_layers=1,
            model_dim=32,
            num_heads=4,
            mlp_mult=2.0,
            tie_output_head=True,
        )
        model = td.DiffusionLM(cfg)
        xt = torch.full((2, cfg.seq_len), cfg.mask_id)
        sigma = torch.full((2,), 0.4)
        logits = model.forward_logits(xt, sigma)
        self.assertEqual(logits.shape, (2, cfg.seq_len, cfg.total_vocab))
        log_probs = model.subs_log_probs(xt, sigma)
        self.assertTrue(torch.all(log_probs[..., cfg.mask_id] < -1e5))
        real_logsum = torch.logsumexp(log_probs[..., : cfg.vocab_size], dim=-1)
        self.assertTrue(torch.allclose(real_logsum, torch.zeros_like(real_logsum), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
