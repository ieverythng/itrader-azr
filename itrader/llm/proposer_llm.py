"""
LLM-based task proposer using Transformers + bitsandbytes for 4-bit quantization.

Generates TaskSpec JSON from market summaries without requiring training loop integration.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from itrader.proposers.base import BaseProposer
from itrader.proposers._shared import parse_json_with_repair, schema_hint as shared_schema_hint


class ProposerLLM(BaseProposer):
    """
    Wrapper around Transformers model with 4-bit quantization for efficient inference.
    """

    def __init__(self, model_id: str = "Qwen2.5-7B-Instruct", device: Optional[str] = None):
        """
        Args:
            model_id: HuggingFace model identifier
            device: "cuda" or "cpu" (auto-detected if None)
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Configure 4-bit quantization (only on CUDA, and only if bitsandbytes available)
        quantization_config = None
        if self.device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            except Exception:
                # Fallback to full precision if bitsandbytes not available
                print("Warning: bitsandbytes not available, loading model in full precision")
                quantization_config = None

        # Load tokenizer and model
        print(f"Loading {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
        self.model.eval()
        print(f"Model loaded successfully.")

    def generate_tasks(
        self,
        market_summary: str,
        n: int,
        schema_hint: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> List[Dict[str, Any]]:
        """
        Generate n TaskSpec dicts from market summary.

        Args:
            market_summary: Text description of market conditions/context
            n: Number of tasks to generate
            schema_hint: Optional schema description (if None, uses default)
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate

        Returns:
            List of TaskSpec dicts (may be empty if parsing fails)
        """
        if schema_hint is None:
            schema_hint = shared_schema_hint()

        prompt = f"""You are a trading task generator. Given the market summary, generate {n} different task specifications as a JSON array.

Market Summary: {market_summary}

Task Schema: {schema_hint}

Generate {n} tasks as a JSON array of objects. Each object must be a valid TaskSpec. Return ONLY the JSON array, no additional text."""

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract JSON (remove prompt part)
        json_text = generated_text[len(prompt):].strip()

        # Parse JSON with repair
        tasks = self._parse_json_with_repair(json_text, n)
        return tasks

    def propose_tasks(self, market_summary: str, n: int, **kwargs: Any) -> List[Dict[str, Any]]:
        return self.generate_tasks(market_summary=market_summary, n=n, **kwargs)

    def _parse_json_with_repair(self, text: str, expected_n: int) -> List[Dict[str, Any]]:
        return parse_json_with_repair(text, expected_n)
