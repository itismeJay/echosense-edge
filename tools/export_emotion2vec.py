"""Export emotion2vec_plus_base (official FunASR checkpoint) to ONNX.

Run once on a Mac (needs torch, already in venv/); copy the .onnx to the Pi.
The runtime (model/emotion_recognizer.py) needs only onnxruntime + numpy.

    curl -L -o models/emotion2vec_plus_base_src/model.pt \
      https://huggingface.co/emotion2vec/emotion2vec_plus_base/resolve/main/model.pt
    pip install --target /tmp/onnx-export onnx   # export-time only, not a runtime dep
    PYTHONPATH=/tmp/onnx-export venv/bin/python tools/export_emotion2vec.py   # fp32, 373 MB
    PYTHONPATH=/tmp/onnx-export venv/bin/python tools/export_emotion2vec.py --int8  # + 117 MB int8

fp32 matches FunASR's reference to 3e-5 (checked on ~/asr-bench/clips) and is
the default; the int8 copy is only for a box short on RAM.

This is an inference-only port of FunASR's Emotion2vec (MIT, data2vec 2.0
audio encoder, post-norm blocks, ALiBi attention). Parameter names match the
checkpoint, so the weights load strictly. The ONNX graph takes raw 16 kHz mono
float32 audio [1, samples] and returns softmax scores over the 9 tokens.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "emotion2vec_plus_base_src"
OUT = ROOT / "models" / "emotion2vec_plus_base.onnx"

EMBED = 768
HEADS = 12
EXTRA_TOKENS = 10
CONV_SPEC = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
POS_DEPTH = 5
POS_K = max(3, 95 // POS_DEPTH)  # 19
POS_GROUPS = 16


class _T(nn.Module):  # TransposeLast
    def forward(self, x):
        return x.transpose(-2, -1)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(EMBED, EMBED * 4)
        self.fc2 = nn.Linear(EMBED * 4, EMBED)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMBED, EMBED * 3)
        self.proj = nn.Linear(EMBED, EMBED)

    def forward(self, x, alibi):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, HEADS, c // HEADS).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * (c // HEADS) ** -0.5) @ k.transpose(-2, -1) + alibi
        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, c))


class _Block(nn.Module):  # AltBlock, layer_norm_first=False
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED, eps=1e-5)
        self.attn = _Attn()
        self.norm2 = nn.LayerNorm(EMBED, eps=1e-5)
        self.mlp = _Mlp()

    def forward(self, x, alibi):
        r = self.norm1(x + self.attn(x, alibi))
        return self.norm2(r + self.mlp(r))


class _ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(4))
        self.norm = nn.LayerNorm(EMBED, eps=1e-5)


def _conv_block(n_in, n_out, k, stride):
    return nn.Sequential(
        nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
        nn.Identity(),
        nn.Sequential(_T(), nn.LayerNorm(n_out), _T()),
        nn.GELU(),
    )


class _LocalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers, n_in = [], 1
        for dim, k, stride in CONV_SPEC:
            layers.append(_conv_block(n_in, dim, k, stride))
            n_in = dim
        self.conv_layers = nn.ModuleList(layers)


class _PosLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module("0", nn.Conv1d(EMBED, EMBED, POS_K, padding=POS_K // 2, groups=POS_GROUPS))

    def forward(self, x):  # B x C x T; odd kernel, SamePad removes nothing
        x = getattr(self, "0")(x)
        x = F.layer_norm(x.transpose(1, 2), (EMBED,)).transpose(1, 2)
        return F.gelu(x)


class _AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_encoder = _LocalEncoder()
        self.project_features = nn.Sequential(_T(), nn.LayerNorm(512), nn.Linear(512, EMBED))
        self.relative_positional_encoder = nn.Sequential(
            _T(), *[_PosLayer() for _ in range(POS_DEPTH)], _T(),
        )
        self.extra_tokens = nn.Parameter(torch.zeros(1, EXTRA_TOKENS, EMBED))
        self.alibi_scale = nn.Parameter(torch.ones(1, 1, HEADS, 1, 1))
        self.context_encoder = _ContextEncoder()


def _alibi_slopes(n):
    def power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]

    if math.log2(n).is_integer():
        return power_of_2(n)
    closest = 2 ** math.floor(math.log2(n))
    return power_of_2(closest) + _alibi_slopes(2 * closest)[0::2][: n - closest]


class Emotion2vecOnnx(nn.Module):
    def __init__(self, vocab_size=9):
        super().__init__()
        self.modality_encoders = nn.ModuleDict({"AUDIO": _AudioEncoder()})
        self.blocks = nn.ModuleList(_Block() for _ in range(8))
        self.proj = nn.Linear(EMBED, vocab_size)
        self.register_buffer(
            "slopes", torch.tensor(_alibi_slopes(HEADS)).view(1, HEADS, 1, 1), persistent=False,
        )

    def forward(self, audio):  # [1, samples] float32, 16 kHz
        x = (audio - audio.mean(dim=1, keepdim=True)) / torch.sqrt(
            audio.var(dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        enc = self.modality_encoders["AUDIO"]
        x = x.unsqueeze(1)
        for conv in enc.local_encoder.conv_layers:
            x = conv(x)
        x = enc.project_features(x)
        x = x + enc.relative_positional_encoder(x)
        t = x.shape[1]
        pos = torch.arange(t, device=x.device, dtype=x.dtype)
        dist = -(pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        alibi = self.slopes * dist.view(1, 1, t, t)
        alibi = alibi * enc.alibi_scale.clamp_min(0).squeeze(0)
        alibi = F.pad(alibi, (EXTRA_TOKENS, 0, EXTRA_TOKENS, 0))
        x = torch.cat([enc.extra_tokens.expand(x.shape[0], -1, -1), x], dim=1)
        x = enc.context_encoder.norm(x)
        for blk in enc.context_encoder.blocks:
            x = blk(x, alibi)
        for blk in self.blocks:
            x = blk(x, alibi)
        x = x[:, EXTRA_TOKENS:].mean(dim=1)
        return torch.softmax(self.proj(x), dim=-1)


def load_model(checkpoint=SRC / "model.pt"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state.get("model", state))
    state = {k.removeprefix("d2v_model."): v for k, v in state.items()}
    model = Emotion2vecOnnx(vocab_size=state["proj.weight"].shape[0])
    wanted = set(model.state_dict())
    state = {k: v for k, v in state.items() if k in wanted}
    model.load_state_dict(state, strict=True)
    return model.eval()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=str(SRC / "model.pt"))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--int8", action="store_true", help="also write a dynamic int8 copy")
    args = parser.parse_args(argv)
    model = load_model(args.checkpoint)
    dummy = torch.randn(1, 32000)
    torch.onnx.export(
        model, (dummy,), args.out, input_names=["audio"], output_names=["scores"],
        dynamic_axes={"audio": {1: "samples"}}, opset_version=17, dynamo=False,
    )
    print(f"[EXPORT] wrote {args.out}")
    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        out8 = str(Path(args.out).with_name(Path(args.out).stem + ".int8.onnx"))
        # MatMul-only, per-channel: default dynamic int8 (convs too) flipped
        # 4 of 12 reference labels; this keeps 29/30 but moves the angry
        # score by up to 0.2, so the runtime prefers the fp32 file.
        quantize_dynamic(
            args.out, out8, weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"], per_channel=True,
        )
        print(f"[EXPORT] wrote {out8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
