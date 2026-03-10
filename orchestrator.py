"""
orchestrator.py
===============
Core inference engine that loads all 3 trained models and provides
a unified classification interface.

Components:
  1. RoBERTa classifier (fast 6-class classification)
  2. Open-set gate (temperature scaling + confidence threshold)
  3. Gemma 2B LoRA (deep JSON analysis for uncertain emails)

Usage:
    from orchestrator import EmailClassifier
    clf = EmailClassifier()      # loads all models to GPU
    result = clf.classify(email_text)
"""

import json
import re
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = "/home/aryan/projects/venketeswar/projects/EP"
CLASSIFIER_DIR = f"{BASE_DIR}/models/classifier"
GATE_CONFIG    = f"{BASE_DIR}/models/gate_config.json"
GEMMA_BASE     = "google/gemma-2b-it"
GEMMA_LORA_DIR = f"{BASE_DIR}/models/gemma_lora"

SYSTEM_PROMPT = (
    "You are a cybersecurity email analyst. "
    "Classify the email and return ONLY a JSON object with keys: "
    "label, confidence, reason, indicators, action."
)


class EmailClassifier:
    """Unified email classification engine with fast + deep paths."""

    def __init__(self, classifier_device="cuda:0", gemma_device="cuda:0"):
        """Load all models once. Use cuda:1 for Gemma if you have 2 GPUs."""
        self.classifier_device = classifier_device
        self.gemma_device = gemma_device

        # ── 1. RoBERTa Classifier ─────────────────────────────────────────
        print("🔍 Loading RoBERTa classifier …")
        self.cls_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
        self.cls_model = AutoModelForSequenceClassification.from_pretrained(
            CLASSIFIER_DIR
        ).to(classifier_device).eval()

        with open(f"{CLASSIFIER_DIR}/label_config.json") as f:
            config = json.load(f)
        self.id2label = {int(k): v for k, v in config["id2label"].items()}
        print(f"  Labels: {list(self.id2label.values())}")

        # ── 2. Gate config ────────────────────────────────────────────────
        print("🚦 Loading gate config …")
        with open(GATE_CONFIG) as f:
            gate = json.load(f)
        self.temperature = gate["T"]
        self.threshold = gate["tau"]
        print(f"  T={self.temperature}, tau={self.threshold}")

        # ── 3. Gemma LoRA ─────────────────────────────────────────────────
        print("🤖 Loading Gemma 2B LoRA (4-bit) …")
        self.gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_LORA_DIR)
        self.gemma_tokenizer.pad_token = self.gemma_tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        gemma_device_map = {"": int(gemma_device.split(":")[-1])}
        base_model = AutoModelForCausalLM.from_pretrained(
            GEMMA_BASE,
            quantization_config=bnb_config,
            device_map=gemma_device_map,
            torch_dtype=torch.bfloat16,
        )
        self.gemma_model = PeftModel.from_pretrained(base_model, GEMMA_LORA_DIR)
        self.gemma_model.eval()
        print("✅ All models loaded!\n")

    # ── Fast path: RoBERTa + Gate ─────────────────────────────────────────
    def _classify_fast(self, text: str) -> dict:
        """Run RoBERTa classifier and apply temperature-scaled gate."""
        enc = self.cls_tokenizer(
            text, max_length=512, truncation=True,
            padding=True, return_tensors="pt"
        ).to(self.classifier_device)

        with torch.no_grad():
            logits = self.cls_model(**enc).logits[0]

        # Temperature-scaled softmax
        scaled = F.softmax(logits / self.temperature, dim=-1)
        confidence, pred_id = scaled.max(dim=-1)
        confidence = confidence.item()
        label = self.id2label[pred_id.item()]

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "logits": logits.cpu(),
            "calibrated_probs": scaled.cpu(),
        }

    # ── Deep path: Gemma LoRA ─────────────────────────────────────────────
    def _classify_deep(self, text: str) -> dict:
        """Run Gemma LoRA for structured JSON analysis."""
        # Truncate to avoid token overflow
        if len(text) > 3000:
            text = text[:3000] + " …"

        messages = [
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + text},
        ]
        inp = self.gemma_tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if hasattr(inp, "input_ids"):
            inp = inp["input_ids"]
        inp = inp.to(self.gemma_model.device)

        with torch.no_grad():
            out = self.gemma_model.generate(
                inp, max_new_tokens=300, do_sample=False
            )
        response = self.gemma_tokenizer.decode(
            out[0][inp.shape[1]:], skip_special_tokens=True
        ).strip()

        # Parse JSON from response
        parsed = self._parse_json(response)
        return parsed if parsed else {
            "label": "unknown",
            "confidence": 0.0,
            "reason": response[:500],
            "indicators": [],
            "action": "review",
        }

    def _parse_json(self, text: str) -> dict | None:
        """Try to parse JSON from model output."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    # ── Main entry point ──────────────────────────────────────────────────
    def classify(self, email_text: str) -> dict:
        """
        Classify an email using the 3-component pipeline.

        Returns:
            {
                "label": str,          # clean/spam/phishing/fraud/malware/threat
                "confidence": float,   # 0.0 - 1.0
                "reason": str,         # explanation (deep path only)
                "indicators": list,    # threat indicators (deep path only)
                "action": str,         # deliver/junk_folder/quarantine/block/review
                "path": str,           # "fast" or "deep"
            }
        """
        # Step 1: Fast classification
        fast = self._classify_fast(email_text)
        label = fast["label"]
        confidence = fast["confidence"]

        # Step 2: Gate decision
        if confidence >= self.threshold:
            # Fast path — high confidence
            action_map = {
                "clean": "deliver",
                "spam": "junk_folder",
                "phishing": "quarantine",
                "fraud": "quarantine",
                "malware": "block",
                "threat": "block",
            }
            return {
                "label": label,
                "confidence": confidence,
                "reason": f"High-confidence classification by encoder ({confidence:.1%})",
                "indicators": [],
                "action": action_map.get(label, "review"),
                "path": "fast",
            }
        else:
            # Deep path — uncertain, run Gemma
            deep = self._classify_deep(email_text)
            deep["path"] = "deep"
            # Ensure all required keys exist
            deep.setdefault("reason", "")
            deep.setdefault("indicators", [])
            deep.setdefault("action", "review")
            deep.setdefault("confidence", 0.0)
            deep.setdefault("label", label)  # fallback to classifier label
            return deep
