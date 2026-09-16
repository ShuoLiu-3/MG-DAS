from __future__ import annotations

from contextlib import ExitStack
import numpy as np
import torch

from .dataio import MultipleChoiceExample
from .intervention import intervention_hooks, reference_scale_hooks, resolve_module
from .types import BBQExample, InterventionConfig


def _torch_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


class HuggingFaceRunner:
    def __init__(
        self,
        model_path: str,
        dtype: str,
        layer_module_template: str,
        token_scope: str,
        prompt_format: str,
        device_map: str = "auto",
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("transformers is required for model execution") from error
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=_torch_dtype(dtype),
            device_map=device_map,
        )
        self.model.eval()
        self.layer_module_template = layer_module_template
        self.token_scope = token_scope
        self.prompt_format = prompt_format

    @property
    def input_device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def build_bbq_prompt(self, example: BBQExample) -> str:
        if self.prompt_format != "original":
            raise ValueError("the paper protocol currently supports prompt_format=original")
        choices = "\n".join(f" {choice}" for choice in example.choices)
        return (
            f"Context: {example.context}\nQuestion: {example.question}\n"
            f"Choices:\n{choices}\nAnswer:\n"
        )

    def _conditional_mean_log_probability(
        self, prompt: str, continuation: str, config: InterventionConfig
    ) -> float:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = self.tokenizer.encode(prompt + continuation, add_special_tokens=True)
        prefix_length = 0
        for prompt_token, full_token in zip(prompt_ids, full_ids):
            if prompt_token != full_token:
                break
            prefix_length += 1
        if prefix_length == len(full_ids):
            raise ValueError("continuation produced no scoreable tokens")
        if prefix_length < max(1, len(prompt_ids) - 2):
            raise ValueError("prompt/continuation tokenization boundary changed unexpectedly")
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.input_device)
        with torch.no_grad():
            if config.cardinality == 0:
                logits = self.model(input_ids=input_ids, use_cache=False).logits.float()
            else:
                layers = tuple(atom.layer for atom in config.atoms)
                with reference_scale_hooks(
                    self.model, layers, self.layer_module_template, self.token_scope
                ) as scales:
                    self.model(input_ids=input_ids, use_cache=False)
                if set(scales) != set(layers):
                    raise RuntimeError("failed to capture every unperturbed reference scale")
                with intervention_hooks(
                    self.model,
                    config,
                    self.layer_module_template,
                    self.token_scope,
                    scales,
                ):
                    logits = self.model(input_ids=input_ids, use_cache=False).logits.float()
        continuation_ids = input_ids[0, prefix_length:]
        prediction_logits = logits[0, prefix_length - 1 : -1, :]
        log_probabilities = torch.log_softmax(prediction_logits, dim=-1)
        token_scores = log_probabilities.gather(1, continuation_ids[:, None]).squeeze(1)
        return float(token_scores.mean().item())

    def score_bbq(
        self, examples: list[BBQExample], config: InterventionConfig
    ) -> np.ndarray:
        scores = np.empty((len(examples), 3), dtype=np.float64)
        for row_index, example in enumerate(examples):
            prompt = self.build_bbq_prompt(example)
            for choice_index, choice in enumerate(example.choices):
                scores[row_index, choice_index] = self._conditional_mean_log_probability(
                    prompt, choice + "\n", config
                )
        return scores

    def score_multiple_choice(
        self, examples: list[MultipleChoiceExample], config: InterventionConfig
    ) -> np.ndarray:
        max_choices = max(len(example.choices) for example in examples)
        scores = np.full((len(examples), max_choices), -np.inf, dtype=np.float64)
        for row_index, example in enumerate(examples):
            for choice_index, choice in enumerate(example.choices):
                scores[row_index, choice_index] = self._conditional_mean_log_probability(
                    example.prompt, choice, config
                )
        return scores

    def extract_bbq_activations(
        self, examples: list[BBQExample], layers: list[int]
    ) -> dict[int, np.ndarray]:
        collected: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        for example in examples:
            prompt = self.build_bbq_prompt(example)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.input_device)
            captured: dict[int, torch.Tensor] = {}
            with ExitStack() as stack:
                for layer in layers:
                    module = resolve_module(self.model, self.layer_module_template.format(layer=layer))

                    def hook(_module, _inputs, output, layer_index=layer):
                        tensor = output[0] if isinstance(output, tuple) else output
                        captured[layer_index] = tensor[:, -1, :].detach().float().cpu()

                    handle = module.register_forward_hook(hook)
                    stack.callback(handle.remove)
                with torch.no_grad():
                    self.model(input_ids=input_ids, use_cache=False)
            if set(captured) != set(layers):
                raise RuntimeError("failed to capture every requested layer")
            for layer in layers:
                collected[layer].append(captured[layer].numpy()[0])
        return {layer: np.stack(rows).astype(np.float32) for layer, rows in collected.items()}


def predictions_from_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2:
        raise ValueError("scores must be a two-dimensional matrix")
    return values.argmax(axis=1)


def multiple_choice_accuracy(scores: np.ndarray, examples: list[MultipleChoiceExample]) -> float:
    predictions = predictions_from_scores(scores)
    labels = np.asarray([example.label_index for example in examples])
    return 100.0 * float(np.mean(predictions == labels))
