import unittest

import torch

import train_gpt_mps_experiment as mps


class UniversalRecurrenceTests(unittest.TestCase):
    def test_universal_mode_reuses_encoder_and_decoder_blocks(self):
        model = mps.GPT(
            vocab_size=32,
            num_layers=6,
            model_dim=16,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.01,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            universal_shared_blocks=True,
        )

        self.assertEqual(len({id(block) for block in model.blocks}), 2)
        self.assertIs(model.blocks[0], model.blocks[1])
        self.assertIs(model.blocks[3], model.blocks[5])

    def test_zero_step_embedding_is_transparent(self):
        shared = mps.GPT(
            vocab_size=32,
            num_layers=4,
            model_dim=16,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.01,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            universal_shared_blocks=True,
            universal_step_embedding=True,
        )
        no_step = mps.GPT(
            vocab_size=32,
            num_layers=4,
            model_dim=16,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.01,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            universal_shared_blocks=True,
            universal_step_embedding=False,
        )
        no_step.load_state_dict(shared.state_dict(), strict=False)
        x = torch.randint(0, 32, (2, 6))
        y = torch.randint(0, 32, (2, 6))

        self.assertTrue(torch.allclose(shared(x, y), no_step(x, y), atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
