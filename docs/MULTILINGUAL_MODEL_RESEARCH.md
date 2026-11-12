# Multilingual sentence-model research

Research date: 2026-11-09. This note compares model artifacts only; no model or
dependency was downloaded. It does not establish that any candidate understands
Cebuano/Bisaya in EchoSense's classroom-safety domain.

## Recommendation

Test `MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli` first, in the existing
disabled, shadow-only semantic path. It is the best practical starting point
because it is already an NLI classifier, has an MIT license, exposes bounded
entailment/neutral/contradiction scores instead of generated prose, and is much
smaller than XLM-R base. Its card reports 0.750 mean XNLI accuracy over the 15
XNLI languages, versus 0.808 for mDeBERTa; the tradeoff is lower expected
accuracy for lower CPU/RAM cost. The one-file FP32 weights are 471 MB, with
about 22 MB of tokenizer assets. [MiniLM model card and results](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli),
[MiniLM repository files](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli/tree/main).

Use `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` as the accuracy-oriented
alternative if MiniLM's held-out errors justify its higher runtime cost. It is
also classifier-ready and MIT-licensed, and its card reports 0.808 mean XNLI
accuracy. Its safetensors file is 558 MB plus about 21 MB of tokenizer assets,
but the card also warns that mDeBERTa does not support FP16. On an ARM CPU that
can imply FP32 execution and materially more working memory than the stored
weight file; only an on-Pi measurement is decision-grade. [mDeBERTa model card](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli),
[mDeBERTa repository files](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli/tree/main).

This recommendation is deliberately provisional. Neither model has published
Cebuano, Tagalog, code-switched, bullying, threat, or classroom-context scores.
It is a recommendation about which artifact to evaluate first, not a capability
claim.

## Evidence standard for the target languages

- XNLI evaluates English plus 14 languages; neither Cebuano nor Tagalog is in
  its language table. MiniLM's and mDeBERTa's published NLI numbers therefore
  do not demonstrate either target language. [XNLI paper](https://arxiv.org/abs/1809.05053)
- Multilingual MiniLM uses the XLM-R tokenizer and was distilled from XLM-R.
  Meta's official XLM-R language list includes Filipino but does not list
  Cebuano. Tokenization is not evidence of sentence understanding, and Filipino
  pretraining is not Tagalog safety-classification evaluation. [Microsoft
  MiniLM documentation](https://github.com/microsoft/unilm/blob/master/minilm/README.md),
  [Meta XLM-R language list](https://github.com/facebookresearch/XLM/blob/main/README.md)
- FLORES-200 includes both `ceb_Latn` and `tgl_Latn`, but it is a translation
  benchmark, not a harassment/intent benchmark. It can test representation
  alignment, not establish EchoSense accuracy. [FLORES-200 language list](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)
- LASER3 explicitly ships a Cebuano-focused encoder, while LASER2 covers
  Tagalog; the maintainers also report experience with code-switched sentences.
  That is stronger representation evidence than a generic “multilingual” label,
  but LASER is an embedding toolkit, not a ready-made EchoSense classifier, and
  its current package brings PyTorch, fairseq, SentencePiece, and Moses-related
  dependencies. The repository was archived in 2025. [LASER3 language list](https://github.com/facebookresearch/LASER/blob/main/nllb/README.md),
  [LASER overview and code-switching statement](https://github.com/facebookresearch/LASER),
  [LASER package dependencies](https://github.com/facebookresearch/LASER/blob/main/pyproject.toml)
- SONAR explicitly covers the NLLB/FLORES languages, including Cebuano and
  Tagalog, but its text encoder weights are non-commercially licensed and it is
  not a classifier. That license and footprint make it a poor deployment
  candidate here despite the language coverage. [SONAR languages](https://github.com/facebookresearch/SONAR#supported-languages-and-download-links),
  [SONAR model licensing](https://github.com/facebookresearch/SONAR/blob/main/LICENSE.md)

No primary source found in this review reports Cebuano classroom-safety
classification results for any candidate. EchoSense must use consenting-adult,
locally held-out examples before making any Cebuano capability statement.

## Candidate comparison

| Candidate                     | Published evidence and license                                                                                                                                | Download / runtime implications                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Fit for EchoSense                                                                                                                                                                                                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Multilingual MiniLMv2 L12 NLI | 0.1B parameters; XNLI/MNLI classifier; 0.750 mean XNLI; MIT                                                                                                   | 471 MB FP32 safetensors plus tokenizer; Transformers/PyTorch/SentencePiece for the reference path. Estimated Pi working set: roughly 0.7–1.3 GB including runtime, tokenizer, activations, and allocator overhead; measure with Whisper and YAMNet active.                                                                                                                                                                                                                                                                                                            | **First test.** Direct zero-shot NLI, bounded output, favorable size/quality balance. Cebuano and Tagalog remain unverified.                                                                                                                                                                              |
| mDeBERTa-v3-base NLI          | 0.3B parameters; XNLI/MNLI classifier; 0.808 mean XNLI; MIT                                                                                                   | 558 MB stored weights plus tokenizer; Transformers/PyTorch/SentencePiece. FP16 warning increases CPU memory risk. Estimated Pi working set: roughly 1.3–2.5 GB if weights are expanded to FP32; must measure.                                                                                                                                                                                                                                                                                                                                                         | **Accuracy alternative.** Better published XNLI result, but slower/heavier and still no target-language evaluation.                                                                                                                                                                                       |
| XLM-R base                    | Pretrained multilingual encoder; 250M parameters in Meta's release; MIT                                                                                       | 1.12 GB safetensors, plus tokenizer; requires a task-specific classification head and held-out training split. [Files](https://huggingface.co/FacebookAI/xlm-roberta-base/tree/main), [Meta documentation](https://github.com/facebookresearch/fairseq/blob/main/examples/xlmr/README.md)                                                                                                                                                                                                                                                                             | Not first: substantially larger and not classifier-ready. Official list includes Filipino, not Cebuano.                                                                                                                                                                                                   |
| Gemma 3 270M IT               | Generative 270M model, 32K context; 140+ language family claim; Gemma Terms, not an OSI license                                                               | 536 MB BF16 weights plus about 39 MB tokenizer assets; autoregressive decoding and strict output parsing add latency/failure modes. [Files](https://huggingface.co/google/gemma-3-270m-it/tree/main), [terms](https://ai.google.dev/gemma/terms)                                                                                                                                                                                                                                                                                                                      | Not first: its model card publishes no multilingual score for the 270M variant, and says content-safety evaluation used English prompts only. It may be useful after task-specific fine-tuning, not as unvalidated prompt-only evidence. [Model card](https://ai.google.dev/gemma/docs/core/model_card_3) |
| EmbeddingGemma 2, text-only   | 270M text path of a 740M modular embedding model; Apache-2.0; intended for classification and semantic similarity; 61.36 aggregate multilingual MTEB-v2 score | The full BF16 checkpoint is 1.49 GB, but Google reports about 191 MB active RAM for quantized text-only weights on Pixel 11 Pro. That number is not a Pi result. Reference path needs recent Transformers/SentenceTransformers/PyTorch; a LiteRT INT4/INT8 path would need a separately approved bundle and compatibility work. [Model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2), [files](https://huggingface.co/google/embeddinggemma-2/tree/main), [edge report](https://developers.googleblog.com/google-ai-edge-with-embeddinggemma-2/) | Strong newer edge candidate, but it produces embeddings, not calibrated class probabilities. Label-prototype or trained-head classification must be validated independently. The published 100+ language statement does not identify or score Cebuano.                                                    |
| multilingual-e5-small         | Roughly 0.1B multilingual embedding model; MIT                                                                                                                | 471 MB safetensors plus tokenizer; SentenceTransformers/PyTorch; needs a trained head or prototype scheme. [Files](https://huggingface.co/intfloat/multilingual-e5-small/tree/main)                                                                                                                                                                                                                                                                                                                                                                                   | Smaller-established embedding baseline, but not a ready classifier and no published Cebuano safety evaluation.                                                                                                                                                                                            |
| LASER2 + Cebuano LASER3       | BSD-licensed embedding software; explicit Tagalog and Cebuano encoders; 1024-d embeddings                                                                     | Separate language routing/models and older PyTorch/fairseq stack; published repository does not give a compact, single-artifact Pi footprint.                                                                                                                                                                                                                                                                                                                                                                                                                         | Valuable Cebuano representation control in a research evaluation, but operationally awkward for code-switching and not a classifier.                                                                                                                                                                      |

The RAM ranges above are engineering estimates, not measurements or vendor
claims. Actual RSS, warm/cold latency, CPU utilization, temperature, throttling,
and queue impact must be captured on the exact Raspberry Pi 5 image while the
existing ASR and acoustic pipeline is running. Quantized quality must be scored
again; it cannot be assumed equal to FP32.

## Evaluation implications

Use identical frozen examples for AI-off and AI-on. Stratify by English,
Cebuano/Bisaya, Tagalog, and code-switching, and cross those strata with:

- direct and indirect targeted insults;
- explicit and conditional threats, including existing genuine-threat cases;
- indirect belittling and exclusion without profanity;
- neutral conversation and academic discussion that mentions harmful terms;
- consensual/playful teasing and ambiguous laughter;
- ASR variants, incomplete sentences, unfamiliar wording, and repeated turns.

Report confusion counts, false positives, missed detections, precision/recall
(with denominators), accuracy per language and category, warm and cold inference
latency, process peak/RSS delta, CPU time/utilization, and, on Pi, temperature
and throttling. Keep laughter, loudness, and acoustic aggression as separate
supporting observations. They must never be encoded as proof of emotion,
playfulness, threat, or intent.

Do not tune prompts, thresholds, examples, or a classification head on the final
held-out set. A small calibration/development split and a separately frozen test
split are required. Mixed-language examples need their own result row rather
than being folded into whichever language dominates the text.

## Downloads that would require explicit approval

For the recommended first experiment, approve these separately:

1. Model snapshot for
   `MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli`, restricted to
   revision `0d55db361c5f291640208c51ff8c181146aa8eff`:
   `model.safetensors` (470,595,778 bytes, SHA-256
   `47b82b3b1f18a0e4cc5cc80d470d75b3e2278603328b3dd67454b19148d7f85b`),
   `tokenizer.json` (17,082,758 bytes, SHA-256
   `098c131bb4423163db239755e309facaa6850059f850f9f3d88a78344a4b631c`),
   `sentencepiece.bpe.model` (5,069,051 bytes, SHA-256
   `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865`),
   plus `config.json`, `tokenizer_config.json`, and
   `special_tokens_map.json`. Do not fetch the duplicate 471 MB
   `pytorch_model.bin`. The selected files total 492,749,469 bytes before
   filesystem/cache overhead.
2. Python wheels for a pinned, ARM64-compatible inference stack. The prepared
   reference adapter uses the existing PyTorch path. The current Mac venv
   already has `torch==2.14.1`, `tokenizers==0.23.3`, and
   `huggingface-hub==2.2.0`; it does not have Transformers, safetensors, regex,
   Typer, or SentencePiece. For the repository's documented CPython 3.11 Pi
   environment, the proposed top-level pins are `transformers==5.22.2` (13.1
   MB universal wheel), `safetensors==0.8.0` (502.8 kB manylinux ARM64 wheel),
   `regex==2026.11.8` (799,760-byte manylinux ARM64 wheel), `typer==0.28.0`
   (123.3 kB universal wheel), and `sentencepiece==0.2.3` (1.6 MB CPython 3.11
   manylinux ARM64 wheel). Typer's currently resolved transitive set is
   `shellingham==1.5.4`, `rich==15.0.0`, `annotated-doc==0.0.5`,
   `markdown-it-py==4.2.0`, and `mdurl==0.1.2`; Pygments is already installed.
   Those ten proposed wheels total 16,532,795 bytes for CPython 3.11 / Linux
   ARM64, assuming the Pi already matches the audited Torch/tokenizers/hub and
   Pygments versions; verify that assumption on the target before approval.
   The final lock must still be checked against the actual Pi OS/Python before
   approval; do not install an sdist or silently replace the existing
   Torch/tokenizers stack. [Transformers wheel](https://pypi.org/project/transformers/),
   [safetensors ARM64 wheel](https://pypi.org/project/safetensors/),
   [SentencePiece ARM64 wheel](https://pypi.org/project/sentencepiece/),
   [Typer wheel](https://pypi.org/project/typer/),
   [regex wheels](https://pypi.org/project/regex/)
3. Optional later optimization only after the FP32 baseline: conversion tooling
   and/or an ONNX INT8 artifact. Do not download an unverified community
   quantization. Record the source model revision and validate numerical output
   plus held-out accuracy after conversion.

The mDeBERTa alternative is **not** part of the first approval: it would require
its 558 MB safetensors file, approximately 21 MB of tokenizer assets, the same
inference dependencies, and a separate approval. Gemma, EmbeddingGemma, LASER,
and SONAR artifacts are also excluded from the first download request.
