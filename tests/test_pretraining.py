"""CPU correctness checks for stage-one pretraining; no RNA corpus required."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from pretraining.data import MLMCollator, PreparedRNADataset, RNATokenizer, build_line_index
from pretraining.model import EncoderConfig, RNAEncoderForMaskedLM, flash_attn_varlen_func
from train_encoder import learning_rate


ROOT = Path(__file__).resolve().parents[1]


class PretrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def tiny_config(self):
        return EncoderConfig(hidden_size=16, num_layers=2, num_heads=2, intermediate_size=24,
                             max_length=16, hidden_dropout=0, attention_dropout=0)

    def test_mask_labels_padding_and_determinism(self):
        tokenizer = RNATokenizer()
        examples = [{"input_ids": tokenizer.encode("ACGURYN"), "mask_seed": 1},
                    {"input_ids": tokenizer.encode("A"), "mask_seed": 2}]
        batch = MLMCollator(1.0)(examples)
        valid = batch["attention_mask"]
        self.assertTrue(batch["input_ids"][valid].eq(tokenizer.mask_token_id).all())
        self.assertTrue(batch["labels"][~valid].eq(-100).all())
        self.assertTrue(batch["input_ids"][~valid].eq(tokenizer.pad_token_id).all())
        self.assertEqual(batch["labels"][0].tolist(), tokenizer.encode("ACGURYN"))
        random_batch = MLMCollator()(examples)
        self.assertTrue(torch.equal(random_batch["input_ids"], MLMCollator()(examples)["input_ids"]))
        self.assertTrue(random_batch["labels"][1, 0].ne(-100))

    def test_epoch_crop_and_worker_independent_randomness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "rna.txt")
            sequences = ["ACGURYSWKMBDHVN" * 5, "ACGU"]
            path.write_text("\n".join(sequences) + "\n", encoding="ascii")
            build_line_index(path, directory)
            dataset = PreparedRNADataset(path, directory, max_length=16)
            before = dataset[0]
            self.assertEqual(before, dataset[0])
            self.assertEqual(len(before["input_ids"]), 16)
            self.assertEqual(dataset[1]["input_ids"], RNATokenizer().encode("ACGU"))
            dataset.epoch = 1
            self.assertNotEqual(before["mask_seed"], dataset[0]["mask_seed"])
            self.assertNotEqual(before["input_ids"], dataset[0]["input_ids"])
            dataset._source.close()
            dataset._offsets = None

    def test_padding_does_not_change_real_token_embeddings(self):
        model = RNAEncoderForMaskedLM(self.tiny_config()).eval()
        tokens = torch.tensor([[5, 6, 7, 8]])
        padded = torch.tensor([[5, 6, 7, 8, 0, 0, 0]])
        with torch.no_grad():
            clean = model(tokens)["last_hidden_state"]
            padded_output = model(padded)["last_hidden_state"]
        torch.testing.assert_close(clean, padded_output[:, :4], atol=1e-6, rtol=1e-5)
        self.assertEqual(float(padded_output[:, 4:].abs().sum()), 0)

    def test_accumulation_matches_combined_token_mean(self):
        model = RNAEncoderForMaskedLM(self.tiny_config())
        batch = MLMCollator()([
            {"input_ids": RNATokenizer().encode("ACGUACGU"), "mask_seed": 1},
            {"input_ids": RNATokenizer().encode("ACG"), "mask_seed": 2},
        ])
        model(**batch)["loss"].backward()
        reference = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        count = 0
        for row in range(2):
            output = model(**{key: value[row:row + 1] for key, value in batch.items()})
            output["loss_sum"].backward()
            count += output["masked_tokens"].item()
        for parameter, expected in zip(model.parameters(), reference):
            torch.testing.assert_close(parameter.grad / count, expected, atol=2e-6, rtol=1e-4)

    def test_checkpointed_backward_and_export(self):
        model = RNAEncoderForMaskedLM(self.tiny_config(), gradient_checkpointing=True)
        batch = MLMCollator()([{"input_ids": RNATokenizer().encode("ACGURYN"), "mask_seed": 1}])
        loss = model(**batch)["loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = RNAEncoderForMaskedLM.from_pretrained(directory)
            model.eval()
            restored.eval()
            torch.testing.assert_close(model(**batch)["logits"], restored(**batch)["logits"])

    def test_schedule_matches_methods(self):
        settings = json.loads((ROOT / "configs/pretraining/encoder.json").read_text())["training"]
        self.assertAlmostEqual(learning_rate(1, settings), 1e-5)
        self.assertAlmostEqual(learning_rate(10000, settings), 1e-4)
        self.assertAlmostEqual(learning_rate(500000, settings), 1e-6)
        self.assertLess(learning_rate(300000, settings), learning_rate(200000, settings))

    @unittest.skipUnless(torch.cuda.is_available() and flash_attn_varlen_func is not None,
                         "Requires CUDA and flash-attn")
    def test_flash_attention_matches_sdpa_with_padding(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("Requires bf16-capable CUDA GPU")
        config = self.tiny_config()
        config.attention_backend = "flash_attention_2"
        flash_model = RNAEncoderForMaskedLM(config).cuda().eval()
        config = self.tiny_config()
        config.attention_backend = "sdpa"
        sdpa_model = RNAEncoderForMaskedLM(config).cuda().eval()
        sdpa_model.load_state_dict(flash_model.state_dict())
        batch = MLMCollator()([
            {"input_ids": RNATokenizer().encode("ACGUACGU"), "mask_seed": 1},
            {"input_ids": RNATokenizer().encode("ACG"), "mask_seed": 2},
        ])
        batch = {key: value.cuda() for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            flash = flash_model(**batch)
            sdpa = sdpa_model(**batch)
        torch.testing.assert_close(flash["logits"], sdpa["logits"], atol=3e-3, rtol=3e-2)
        flash["loss"].backward()
        sdpa["loss"].backward()
        for a, b in zip(flash_model.parameters(), sdpa_model.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=5e-3, rtol=5e-2)

    def test_training_resume_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            corpus = directory / "rna.txt"
            corpus.write_text("\n".join(["ACGUACGUACGU", "GGGAAACCCUUU", "RYACGUN", "A"] * 3) + "\n")
            config = json.loads((ROOT / "configs/pretraining/encoder.json").read_text())
            config["model"].update(hidden_size=16, num_layers=1, num_heads=2, intermediate_size=24, max_length=16)
            config["training"].update(global_batch_size=4, micro_batch_size=2, max_steps=4, warmup_steps=2,
                                     precision="fp32", num_workers=0, log_every=1, save_every=2, eval_every=2)
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(config))
            command = [sys.executable, str(ROOT / "train_encoder.py"), "--config", str(config_path),
                       "--train_data", str(corpus), "--validation_data", str(corpus), "--device", "cpu"]
            def run(extra):
                result = subprocess.run(command + extra, cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run(["--output_dir", str(directory / "full")])
            run(["--output_dir", str(directory / "resumed"), "--stop_after_steps", "2"])
            run(["--output_dir", str(directory / "resumed"), "--resume", str(directory / "resumed/checkpoint-00000002.pt"), "--num_workers", "1"])
            full = torch.load(directory / "full/checkpoint-00000004.pt", weights_only=False)
            resumed = torch.load(directory / "resumed/checkpoint-00000004.pt", weights_only=False)
            for key, expected in full["model"].items():
                torch.testing.assert_close(resumed["model"][key], expected, rtol=0, atol=0)
            self.assertEqual((full["epoch"], full["batch_cursor"]), (resumed["epoch"], resumed["batch_cursor"]))
            self.assertTrue(torch.equal(full["rng_states"][0]["torch"], resumed["rng_states"][0]["torch"]))


if __name__ == "__main__":
    unittest.main()
