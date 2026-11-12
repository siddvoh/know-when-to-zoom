# Know When to Zoom

Code and data for *Know When to Zoom: How Small Is Too Small for Vision-Language Models?* by Siddharth Vohra and Min Xu, accepted to the SENSE workshop at WACV 2027. The paper and a plain-language walkthrough are at https://siddvoh.com/know-when-to-zoom/.

Online services shrink a large image, or squeeze it into a fixed grid of tokens, before a vision-language model sees it. Seven models read an eye exam of random five-character codes printed at ten text sizes on gray images from 1024 by 768 to 8000 by 6000 pixels. They then answered 601 multiple-choice questions about small objects in large photos from V*Bench, HR-Bench 8K and MME-RealWorld-Lite, with each photo sent as is, at Gemini's highest setting, as tiles, or as a two-step zoom. This repository holds all 31,053 answers and the code that turns them into the paper's results. Rebuilding the results calls no model.

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

## Rebuilding the results

With Python 3.11 or newer:

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python code/analysis.py
python code/render.py eye
python code/figures.py
```

This takes about five minutes, most of it spent drawing the eye-exam images. `analysis.py` writes `data/results/values.json`, `render.py eye` draws the eye-exam images that the first figure shows, and `figures.py` writes the five figures to `figures/`. The copies in this repository are their output. Font rendering can differ slightly between systems, so a redrawn eye-exam image may not match the SHA-256 logged in `data/images.jsonl`.

## Data

`data/items/eye.jsonl` holds the eye-exam items: the code, its cap height in pixels, the canvas size and the repeat. `data/items/photos.jsonl` holds the 601 photo questions: the benchmark and subset, the source image, the question, its options, the answer and, for V*Bench, the target boxes. The prompts in `code/items.py` add a "cannot tell" option to every question.

`data/calls/<model>.jsonl` has one record per call. `part` is `eye`, `photos` or `zoom` (the second call of the two-step zoom), and `setting` is the resolution setting for the eye exam or, for photos, `as_is`, `max` (Gemini's highest setting), `tile2`, `tile3`, `zoom1` or `zoom2`. Each record lists the images sent with their SHA-256, and keeps the full reply in `text`, the parsed answer in `parsed`, the reported token counts in `tokens` and the reported model version. `status` is `ok`, `rejected` when the service refused the request, or `unparsed` when the reply held no valid answer. A `zoom2` record also keeps the box the model named, whether it came from the model or fell back to the whole photo, and the crop that was sent. Claude's standard setting is called `old-tier` in the code and data.

`data/images.jsonl` lists every image sent, with its size, format, JPEG quality and SHA-256. `data/sources.json` lists the benchmark files with their revisions and SHA-256.

## Code

`code/zoomcheck.py` holds each service's resize and rejection rules and the zoom check. `code/items.py` defines the models, the settings, the prompts, the eye-exam items and the photo questions. `code/render.py` draws every image sent to a model: the eye-exam canvases, the photos, the tiles and the zoom crops. `code/providers.py` builds the request for each model and reads its reply, and `code/run.py` sends the calls and stores one record per call. `code/analysis.py` computes the paper's results, `code/figures.py` draws the figures, and `code/download.py` fetches the benchmarks.

## Running the study again

`run.py` calls paid models, so it needs API keys in `.env`:

```
cp .env.example .env
python code/download.py
python code/items.py
python code/render.py eye photos
python code/run.py eye
python code/run.py photos
python code/run.py zoom
```

Start from an empty `data/calls` and `data/images.jsonl`, since `run.py` skips every call that already has an answer. The models sample their replies, so a new run follows the same procedure but gives new answers.

## Citation

```
@inproceedings{vohra2027zoom,
  title     = {Know When to Zoom: How Small Is Too Small for Vision-Language
               Models?},
  author    = {Vohra, Siddharth and Xu, Min},
  booktitle = {Sensing via Agentic Reasoning Systems (SENSE) Workshop at
               WACV 2027},
  year      = {2027},
  address   = {Orlando, Florida, USA},
  note      = {To appear},
  url       = {https://siddvoh.com/know-when-to-zoom/}
}
```

## License

The code is under the MIT license in `LICENSE`. The benchmark material keeps its own terms, which are listed in `NOTICE.txt`. No benchmark images are included; `code/download.py` fetches them.
