# Know When to Zoom: How Small Is Too Small for Vision-Language Models?

Online services shrink a large image, or squeeze it into a fixed grid of tokens, before a vision-language model sees it. This repository holds the code and every model answer behind our study of how much small detail survives that step.

An eye exam prints random five-character codes at ten text sizes on gray images from 1024 by 768 to 8000 by 6000 pixels, and asks seven models to read them. We then ask 601 multiple-choice questions about small objects in large photos from V* Bench, HR-Bench 8K and MME-RealWorld-Lite, sending each photo as is, at Gemini's highest setting, as tiles, or as a two-step zoom.

## Models

| Tier | Model | Maker | Model ID |
|---|---|---|---|
| Small | Claude Haiku 5.5 | Anthropic | `claude-haiku-5-5` |
| Small | Gemini 3.5 Flash-Lite | Google | `gemini-3.5-flash-lite` |
| Small | GPT-6 Luna | OpenAI | `gpt-6-luna` |
| Mid | Claude Sonnet 5.5 | Anthropic | `claude-sonnet-5-5` |
| Mid | Gemini 3.8 Flash | Google | `gemini-3.8-flash` |
| Mid | GPT-6.1 Sol | OpenAI | `gpt-6.1-sol` |
| Reference | Qwen3.8-27B | Qwen | `qwen3.8-27b` |

## Reproducing the results

With Python 3.11 or newer:

    pip install -r requirements.txt
    python code/analysis.py
    python code/render.py eye
    python code/figures.py

`analysis.py` computes the paper's main results from the answers in `data/calls` and writes them to `data/results/values.json`. `render.py eye` draws the eye-exam images used in the first figure, and `figures.py` writes the figures to `figures/`. None of these call a model.

## Running the study again

    cp .env.example .env
    python code/download.py
    python code/items.py
    python code/render.py eye photos
    python code/run.py eye
    python code/run.py photos
    python code/run.py zoom

`.env` holds the API keys. Start from an empty `data/calls` and `data/images.jsonl`, since `run.py` skips every call that already has an answer.

## Contents

    code/zoomcheck.py       the service rules and the zoom check
    code/items.py           the models, prompts, eye-exam items and photo questions
    code/render.py          every image sent to a model
    code/providers.py       the requests to each model
    code/run.py             sends the calls and stores the answers
    code/analysis.py        every number in the paper
    code/figures.py         the figures
    code/download.py        the benchmark downloads
    data/items/             the eye-exam items and the photo questions
    data/images.jsonl       every image sent, with its size and SHA-256
    data/calls/<model>      one record per call, with the images sent and the full answer
    data/results/           values.json
    data/sources.json       the benchmark files, with revisions and SHA-256
    figures/                the figures in the paper

## Data terms

The code is under the MIT license in `LICENSE`. The benchmarks keep their own terms, listed in `NOTICE.txt`. No benchmark images are included; `code/download.py` fetches them.

## Citation

    @misc{vohra2026zoom,
      title  = {Know When to Zoom: How Small Is Too Small for Vision-Language Models?},
      author = {Siddharth Vohra and Min Xu},
      year   = {2026}
    }
