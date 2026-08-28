import argparse
import json
import os
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LayerPerturber:


    def __init__(
        self,
        model: torch.nn.Module,
        vector: torch.Tensor,
        *,
        scale: float = 1.0,
        layer_idx: Optional[int] = None,
        all_layers: bool = False,
        token_mode: str = "all",
        debug: bool = False,
    ) -> None:
        self.model = model
        self.scale = float(scale)
        self.layer_idx = layer_idx
        self.all_layers = all_layers
        self.token_mode = token_mode.lower()
        self.debug = debug

        p = next(model.parameters())
        vec = torch.as_tensor(vector, dtype=p.dtype, device=p.device)
        if vec.ndim != 1:
            raise ValueError("vector must be 1-D")
        hidden = getattr(model.config, "hidden_size", None)
        if hidden and vec.numel() != hidden:
            raise ValueError(f"Vector dim {vec.numel()} != hidden_size {hidden}")
        self.vector = vec

        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _possible_layer_attrs(self) -> List[str]:
        return [
            "transformer.h",
            "encoder.layer",
            "model.layers",
            "gpt_neox.layers",
            "block",
        ]

    def _get_layers(self):
        for path in self._possible_layer_attrs():
            cur = self.model
            ok = True
            for part in path.split('.'):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "__getitem__"):
                return path, cur
        raise ValueError("Could not locate transformer layer list on model")

    def _hook_fn(self, module, ins, out):
        steer = self.scale * self.vector

        def add_vec(t: torch.Tensor) -> torch.Tensor:
            if t.ndim == 3:
                if self.token_mode == "bos":
                    t2 = t.clone()
                    if t2.shape[1] > 0:
                        t2[:, 0, :] += steer.to(t2.device)
                    return t2
                else:
                    return t + steer.to(t.device)
            elif t.ndim == 2:
                return t + steer.to(t.device)
            return t

        if torch.is_tensor(out):
            return add_vec(out)
        elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            head = add_vec(out[0])
            return (head, *out[1:])
        return out

    def __enter__(self):
        path, layers = self._get_layers()
        if self.all_layers:
            for i in range(len(layers)):
                h = layers[i].register_forward_hook(self._hook_fn)
                self._handles.append(h)
            if self.debug:
                print(f"[LayerPerturber] Injecting into ALL layers of {path} (n={len(layers)}) with token_mode={self.token_mode}")
        else:
            if self.layer_idx is None:
                raise ValueError("layer_idx must be set when all_layers is False")
            if not (-len(layers) <= self.layer_idx < len(layers)):
                raise IndexError("layer_idx out of range")
            h = layers[self.layer_idx].register_forward_hook(self._hook_fn)
            self._handles.append(h)
            if self.debug:
                print(f"[LayerPerturber] Injecting into {path}[{self.layer_idx}] with token_mode={self.token_mode}")
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def load_model_and_tokenizer(model_id: str):
    """
    Unified model loading logic.
    Handles GPT-OSS / quantized models with torch_dtype="auto" to avoid dtype conflicts.
    Sets pad_token and padding_side='left' for decoder-only models.
    """
    use_gpu = torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if "gpt-oss" in model_id.lower():
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        return tokenizer, model

    if use_gpu:
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.padding_side:
        tokenizer.padding_side = "left"

    return tokenizer, model


def _possible_layer_attrs() -> List[str]:
    return [
        "transformer.h",
        "encoder.layer",
        "model.layers",
        "gpt_neox.layers",
        "block",
    ]


def _locate_blocks(model):
    for path in _possible_layer_attrs():
        cur = model
        ok = True
        for part in path.split('.'):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and hasattr(cur, "__getitem__"):
            return cur
    raise ValueError("Could not locate transformer blocks on model")


def _get_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


"""

To be released after the acceptance.

"""