"""Real image-capable model adapter; labels are supplied only to the scorer."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from fabric_inspector.common import now, resolve_under, write_json, environment
from fabric_inspector.evaluation.metrics import metrics

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
# Immutable publisher commit verified from the model repository on 2026-10-11.
MODEL_REVISION = "7e3e67edbbed1bf9888184d9df282b700a323964"
PROMPTS = [
    "Inspect this fabric patch for a visible manufacturing defect. Reply with only one label: defective or defect_free.",
    "Classify this fabric patch. Look for a local disruption, hole, stain or anomalous yarn rather than ordinary repeating weave or lighting. Reply with only one label: defective or defect_free.",
]


def parse_decision(response: str, response_format="json") -> int:
    text = response.strip()
    if response_format == "label":
        # A declared formatting normalisation, never a substring search or an
        # inference from descriptions. Ambiguous/multi-sentence outputs fail.
        matched = re.fullmatch(
            r"(?:(?:the|this) fabric(?: patch)? is )?(defective|defect_free|defect-free|defect free)[.!]?",
            text,
            flags=re.I,
        )
        if not matched:
            raise ValueError("Expected one unambiguous binary label or canonical decision sentence")
        return int(matched.group(1).lower() == "defective")
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"decision"}:
        raise ValueError("Expected exactly one decision field")
    if value["decision"] not in {"defective", "defect_free"}:
        raise ValueError("Invalid binary decision")
    return int(value["decision"] == "defective")


class LocalVLM:
    """SmolVLM 256M local CPU-compatible inference, without external paid APIs."""

    def __init__(self, config: dict):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from fabric_inspector.models.neural import choose_device

        torch.set_num_threads(config.get("cpu_threads", 4))
        self.config, self.device = config, choose_device(config.get("device", "cpu"))
        self.processor = AutoProcessor.from_pretrained(
            config["model_id"], revision=config["revision"], trust_remote_code=False
        )
        self.processor.image_processor.do_image_splitting = config.get("do_image_splitting", False)
        side = config.get("processor_longest_edge", 256)
        self.processor.image_processor.size = {"longest_edge": side}
        self.processor.image_processor.max_image_size = {"longest_edge": side}
        self.processor.image_seq_len = config.get("image_seq_len", 16)
        dtype = torch.bfloat16 if config.get("dtype") == "bfloat16" else torch.float32
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                config["model_id"],
                revision=config["revision"],
                dtype=dtype,
                trust_remote_code=False,
            )
            .to(self.device)
            .eval()
        )
        self.dtype = dtype
        self.resolved_revision = (
            getattr(self.model.config, "_commit_hash", None) or config["revision"]
        )

    def predict_image(self, image: Image.Image, prompt: str) -> dict:
        """The inference interface intentionally has no label, mask, filename or ANN score."""
        import torch

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # Strip PNG metadata; images are passed as pixel content, never revealing paths.
        pixels = Image.fromarray(np.asarray(image.convert("RGB")).copy())
        started = time.perf_counter()
        inputs = self.processor(text=text, images=[pixels], return_tensors="pt").to(
            self.device, dtype=self.dtype
        )
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.config["max_new_tokens"], do_sample=False
            )
        tokens = outputs[0, inputs["input_ids"].shape[-1] :]
        response = self.processor.decode(tokens, skip_special_tokens=True)
        return {
            "raw_response": response,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "input_tokens": int(inputs["input_ids"].numel()),
            "output_tokens": int(tokens.numel()),
            "resolved_revision": self.resolved_revision,
        }


def default_vlm_config(prompt_index=0) -> dict:
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "prompt": PROMPTS[prompt_index],
        "prompt_index": prompt_index,
        "max_new_tokens": 24,
        "do_sample": False,
        "quantisation": "none",
        "dtype": "bfloat16",
        "device": "cpu",
        "cpu_threads": 1,
        "do_image_splitting": False,
        "processor_longest_edge": 256,
        "image_seq_len": 16,
        "response_format": "label",
        "parsing_policy": "Exact defective/defect_free token or one canonical sentence: The/This fabric (patch) is defective/defect-free. Any extra reasoning or ambiguity is unresolved.",
        "max_retries": 1,
        "failure_policy": "Unresolved (-1) counts as an error for its true class; retain every example",
        "image_processor": "Publisher AutoProcessor at the same immutable revision",
        "image_size_note": "Underlying 256x256 pixels preserved; no splitting or upscaling. 16 visual tokens = (256/16 patch size)^2 / 4^2 pixel-shuffle factor. Processor token count explicitly matches vision output.",
        "marginal_api_charge": 0,
        "compute_cost": "Hardware, electricity and CPU time; not measured as money",
    }


def benchmark(
    adapter,
    config: dict,
    rows: list[dict],
    root: Path,
    output: Path,
    partition: str,
    provenance: str,
) -> dict:
    response_log = output.with_suffix(".responses.jsonl")
    if output.exists() or response_log.exists():
        raise FileExistsError("Benchmark output exists; preserve the original evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    response_log.touch(exist_ok=False)
    warmup = []
    for row in rows[:3]:
        with Image.open(resolve_under(root, row["image_path"])) as image:
            warmup.append(adapter.predict_image(image, config["prompt"]))
    predictions, entries = [], []
    for row in rows:
        example_started = time.perf_counter()
        prediction, attempts = -1, []
        with Image.open(resolve_under(root, row["image_path"])) as image:
            clean_image = image.convert("RGB").copy()
        for attempt in range(config["max_retries"] + 1):
            started = time.perf_counter()
            try:
                response = adapter.predict_image(clean_image, config["prompt"])
                attempts.append(response)
                prediction = parse_decision(
                    response["raw_response"], config.get("response_format", "json")
                )
                break
            except Exception as error:
                if len(attempts) <= attempt:
                    attempts.append(
                        {"latency_ms": (time.perf_counter() - started) * 1000, "raw_response": None}
                    )
                attempts[-1]["error"] = f"{type(error).__name__}: {error}"
        entries.append(
            {
                "patch_id": row["patch_id"],
                "image_sha256": row["sha256"],
                "prediction": prediction,
                "attempts": attempts,
                "end_to_end_ms": (time.perf_counter() - example_started) * 1000,
            }
        )
        predictions.append(prediction)
        # Persist each actual response promptly; interrupted runs retain their
        # evidence without repeatedly rewriting the entire growing benchmark.
        with response_log.open("a") as stream:
            stream.write(json.dumps(entries[-1], allow_nan=False) + "\n")
        if len(entries) == 1 or len(entries) % 25 == 0 or len(entries) == len(rows):
            print(f"VLM {partition}: {len(entries)}/{len(rows)} examples attempted", flush=True)
    y = [int(r["label"]) for r in rows]
    summary = {
        "status": "COMPLETE",
        "partition": partition,
        "provenance": provenance,
        "timestamp": now(),
        "config": config,
        "environment": environment(),
        "prompt_sha256": hashlib.sha256(config["prompt"].encode()).hexdigest(),
        "metrics": metrics(y, predictions),
        "examples": entries,
        "latency_scope": "File decode, processor, generation and parsing; includes retries; excludes one-time model loading and response-log writes",
        "batch_size": 1,
        "warmup_examples": len(warmup),
        "warmup_responses": warmup,
        "mean_latency_ms": float(np.mean([e["end_to_end_ms"] for e in entries])),
        "p95_latency_ms": float(np.quantile([e["end_to_end_ms"] for e in entries], 0.95)),
        "failure_rate": float(np.mean(np.array(predictions) == -1)),
        "labels_used_only_for_scoring": True,
        "no_probability_or_auc_claim": True,
    }
    write_json(output, summary)
    return summary
