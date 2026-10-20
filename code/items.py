import base64
import json
import random
import zipfile
from pathlib import Path

import matplotlib
import pyarrow.parquet as pq
from PIL import Image, ImageFont

import config
import zoomcheck

Image.MAX_IMAGE_PIXELS = None

ALPHABET = "ACDEFHJKMNPRTUVWXY3479"
CODE_LENGTH = 5
CAPS = (4, 6, 9, 13, 19, 27, 40, 59, 87, 128)
CANVASES = ((1024, 768), (2048, 1536), (4096, 3072), (8000, 6000))
CODES_PER_CELL = 20
REPEATS = 80
EDGE_MARGIN = 0.10
PADDING = 0.30
RING_RADIUS = 0.04
RING_WIDTH = 0.006
BACKGROUND = (128, 128, 128)
PHOTO_COUNTS = {"vstar": None, "hrbench": (75, 75), "mme-rs": 130, "mme-mon": 130}

EYE_PROMPT = (
    'A red ring marks a small white label in this image. The label shows a code of exactly 5 characters, '
    'using only capital letters and digits.\n'
    'Read the code. If you cannot read it, set "readable" to false.\n'
    'Reply with one JSON object and nothing else:\n'
    '{"code": "<the 5 characters, or an empty string>", "readable": <true or false>, '
    '"confidence": <integer from 0 to 100>}'
)
PHOTO_PROMPT = (
    'Answer the question about the image.\n'
    'Question: {question}\n'
    'Options:\n'
    '{options}\n'
    'X. The image does not show enough detail to tell.\n'
    'Reply with one JSON object and nothing else:\n'
    '{"answer": "<one option letter>", "confidence": <integer from 0 to 100>}'
)
TILE_LINE = ('Image 1 is the whole photo. Images 2 to {n} are enlarged parts of the same photo, in reading order '
             'from top left to bottom right, overlapping slightly.')
ZOOM_LINE = 'Image 1 is the whole photo. Image 2 is an enlarged part of it.'
BOX_PROMPT = (
    'Question about this image: {question}\n'
    'Options: {options}\n'
    'Do not answer yet. Give the region of the image you need to see up close to answer.\n'
    'Give each edge of the region as a fraction from 0 to 1: "left" and "right" are measured from the left '
    'edge of the image as fractions of its width, and "top" and "bottom" are measured from the top edge as '
    'fractions of its height.\n'
    'Reply with one JSON object and nothing else:\n'
    '{"left": <0 to 1>, "top": <0 to 1>, "right": <0 to 1>, "bottom": <0 to 1>}'
)
EDGES = ("left", "top", "right", "bottom")

MODELS = {
    "claude-haiku-5-5": {"tier": "small", "vendor": "anthropic", "model_id": "claude-haiku-5-5",
                         "eye": ["native", "old-tier"], "default": "native",
                         "actions": ["as_is", "tile2", "tile3", "zoom"]},
    "claude-sonnet-5-5": {"tier": "mid", "vendor": "anthropic", "model_id": "claude-sonnet-5-5",
                          "eye": ["native", "old-tier"], "default": "native",
                          "actions": ["as_is", "tile2", "tile3", "zoom"]},
    "gemini-3.5-flash-lite": {"tier": "small", "vendor": "gemini", "model_id": "gemini-3.5-flash-lite",
                              "eye": ["high", "ultra_high"], "default": "default", "max": "ultra_high",
                              "actions": ["as_is", "max", "tile2", "tile3", "zoom"]},
    "gemini-3.8-flash": {"tier": "mid", "vendor": "gemini", "model_id": "gemini-3.8-flash",
                         "eye": ["low", "medium", "high", "ultra_high"], "default": "default", "max": "ultra_high",
                         "actions": ["as_is", "max", "tile2", "tile3", "zoom"]},
    "gpt-6-luna": {"tier": "small", "vendor": "openai", "model_id": "openai/gpt-6-luna", "provider": "openai",
                   "eye": ["auto", "high"], "default": "auto", "actions": ["as_is", "tile2", "tile3", "zoom"]},
    "gpt-6.1-sol": {"tier": "mid", "vendor": "openai", "model_id": "openai/gpt-6.1-sol", "provider": "openai",
                    "eye": ["auto", "high"], "default": "auto", "actions": ["as_is", "tile2", "tile3", "zoom"]},
    "qwen3.8-27b": {"tier": "reference", "vendor": "qwen", "model_id": "qwen/qwen3.8-27b",
                    "provider": "deepinfra/bf16", "eye": ["default"], "default": "default", "actions": ["as_is"]},
}


def font_path():
    return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"


def cap_ratio():
    font = ImageFont.truetype(str(font_path()), 1000)
    left, top, right, bottom = font.getbbox("H", anchor="ls")
    return (bottom - top) / 1000


def label_geometry(code, cap, ratio):
    font = ImageFont.truetype(str(font_path()), cap / ratio)
    left, top, right, bottom = font.getbbox(code, anchor="ls")
    pad = PADDING * cap
    return {"font_size": cap / ratio, "width": right - left + 2 * pad, "height": bottom - top + 2 * pad}


def eye_item(width, height, cap, number, ratio):
    item_id = f"eye-{width}x{height}-c{cap:03d}-{number:02d}"
    rng = random.Random(config.seed_of(item_id))
    code = "".join(rng.choice(ALPHABET) for _ in range(CODE_LENGTH))
    shape = label_geometry(code, cap, ratio)
    margin_x = min(EDGE_MARGIN * width, (width - shape["width"]) / 2)
    margin_y = min(EDGE_MARGIN * height, (height - shape["height"]) / 2)
    x0 = rng.uniform(margin_x, width - margin_x - shape["width"])
    y0 = rng.uniform(margin_y, height - margin_y - shape["height"])
    label = [round(x0), round(y0), round(x0 + shape["width"]), round(y0 + shape["height"])]
    long_edge = max(width, height)
    ring_width = max(1, round(RING_WIDTH * long_edge))
    half_diagonal = 0.5 * ((label[2] - label[0]) ** 2 + (label[3] - label[1]) ** 2) ** 0.5
    radius = max(RING_RADIUS * long_edge, half_diagonal + 2 * ring_width)
    return {"id": item_id, "width": width, "height": height, "cap": cap, "code": code, "label": label,
            "font_size": shape["font_size"],
            "ring": {"center": [(label[0] + label[2]) / 2, (label[1] + label[3]) / 2], "radius": round(radius, 2),
                     "width": ring_width}}


def eye_items():
    ratio = cap_ratio()
    eye = [eye_item(width, height, cap, number, ratio)
           for width, height in CANVASES for cap in CAPS for number in range(CODES_PER_CELL)]
    repeated = set(config.rank([item["id"] for item in eye], "repeat")[:REPEATS])
    for item in eye:
        item["repeat"] = item["id"] in repeated
    return eye


def lettered(options):
    return [{"letter": chr(ord("A") + number), "text": text} for number, text in enumerate(options)]


def vstar_items():
    items = []
    for folder in ("direct_attributes", "relative_position"):
        for path in sorted((config.DOWNLOADS / "vstar" / folder).glob("*.json")):
            spec = json.loads(path.read_text())
            item_id = f"vstar-{folder}-{path.stem}"
            order = config.rank(range(len(spec["options"])), "options", item_id)
            options = lettered([spec["options"][number] for number in order])
            image = path.with_suffix(".jpg")
            width, height = Image.open(image).size
            items.append({"id": item_id, "dataset": "vstar", "subset": folder, "question": spec["question"],
                          "options": options, "answer": options[order.index(0)]["letter"],
                          "source": config.relative(image), "image": f"vstar-{path.stem}",
                          "width": width, "height": height,
                          "targets": [{"name": name, "box": box}
                                      for name, box in zip(spec["target_object"], spec["bbox"])]})
    return items


def hrbench_rows():
    table = pq.read_table(config.DOWNLOADS / "hrbench" / "hr_bench_8k.parquet",
                          columns=["index", "question", "answer", "category", "A", "B", "C", "D", "cycle_category"])
    return [row for row in table.to_pylist() if row["cycle_category"] == 0]


def hrbench_images(indices):
    folder = config.DOWNLOADS / "hrbench" / "images"
    missing = [index for index in indices if not (folder / f"{index}.jpg").exists()]
    if missing:
        parquet = pq.ParquetFile(config.DOWNLOADS / "hrbench" / "hr_bench_8k.parquet")
        for batch in parquet.iter_batches(batch_size=4, columns=["index", "image"]):
            for index, text in zip(batch.column("index").to_pylist(), batch.column("image").to_pylist()):
                if index in missing:
                    config.write_bytes(folder / f"{index}.jpg", base64.b64decode(text))
    return {index: folder / f"{index}.jpg" for index in indices}


def hrbench_items():
    rows = hrbench_rows()
    chosen = []
    for category, count in zip(("single", "cross"), PHOTO_COUNTS["hrbench"]):
        pool = {row["index"]: row for row in rows if row["category"] == category}
        chosen += [pool[index] for index in config.rank(pool, "hrbench", category)[:count]]
    paths = hrbench_images([row["index"] for row in chosen])
    items = []
    for row in chosen:
        item_id = f"hrbench-{row['index']}"
        texts = [row[letter] for letter in "ABCD"]
        correct = "ABCD".index(row["answer"])
        order = config.rank(range(4), "options", item_id)
        options = lettered([texts[number] for number in order])
        path = paths[row["index"]]
        width, height = Image.open(path).size
        items.append({"id": item_id, "dataset": "hrbench", "subset": row["category"], "question": row["question"],
                      "options": options, "answer": options[order.index(correct)]["letter"],
                      "source": config.relative(path), "image": f"hrbench-{config.sha256_file(path)[:16]}",
                      "width": width, "height": height})
    return items


def mme_items():
    archive = zipfile.ZipFile(config.DOWNLOADS / "mme" / "data.zip")
    questions = json.loads(archive.read("data/MME-RealWorld-Lite.json"))
    items = []
    for key, subtask in (("mme-rs", "Remote Sensing"), ("mme-mon", "Monitoring")):
        pool = {row["Question_id"]: row for row in questions
                if row["Task"] == "Perception" and row["Subtask"] == subtask}
        for question_id in sorted(config.rank(pool, key)[:PHOTO_COUNTS[key]]):
            row = pool[question_id]
            options = [{"letter": choice[1], "text": choice[4:].strip()} for choice in row["Answer choices"]]
            member = f"data/imgs/{row['Image']}"
            with archive.open(member) as handle:
                width, height = Image.open(handle).size
            items.append({"id": f"{key}-{question_id.replace('/', '-')}", "dataset": key,
                          "subset": row["Category"], "question": row["Text"], "options": options,
                          "answer": row["Ground truth"], "source": f"data/downloads/mme/data.zip:{member}",
                          "image": f"{key}-{Path(row['Image']).stem}", "width": width, "height": height})
    return items


def option_lines(item):
    return "\n".join(f"{option['letter']}. {option['text']}" for option in item["options"])


def photo_prompt(item, lead=None):
    text = PHOTO_PROMPT.replace("{question}", item["question"]).replace("{options}", option_lines(item))
    return f"{lead}\n{text}" if lead else text


def box_prompt(item):
    options = "; ".join(f"{option['letter']}. {option['text']}" for option in item["options"])
    return BOX_PROMPT.replace("{question}", item["question"]).replace("{options}", options)


def tile_lead(n):
    return TILE_LINE.replace("{n}", str(n + 1))


def call(model, part, setting, item, repeat=1):
    call_id = f"{model}|{part}|{setting}|{item}" + (f"|r{repeat}" if repeat > 1 else "")
    return {"id": call_id, "model": model, "part": part, "setting": setting, "item": item, "repeat": repeat}


def photo_actions(spec, item):
    actions = []
    for action in spec["actions"]:
        if action == "tile3" and max(item["width"], item["height"]) < zoomcheck.TILE3_MIN_EDGE:
            continue
        actions.append("zoom1" if action == "zoom" else action)
    return actions


def planned_calls(eye, photos):
    calls = []
    for model, spec in MODELS.items():
        for setting in spec["eye"]:
            for item in eye:
                calls.append(call(model, "eye", setting, item["id"]))
                if item["repeat"]:
                    calls.append(call(model, "eye", setting, item["id"], 2))
    for model, spec in MODELS.items():
        for item in photos:
            calls += [call(model, "photos", action, item["id"]) for action in photo_actions(spec, item)]
    for model, spec in MODELS.items():
        if "zoom" in spec["actions"]:
            calls += [call(model, "zoom", "zoom2", item["id"]) for item in photos]
    return calls


def load_items():
    return config.read_jsonl(config.ITEMS / "eye.jsonl"), config.read_jsonl(config.ITEMS / "photos.jsonl")


def main():
    eye = eye_items()
    photos = vstar_items() + hrbench_items() + mme_items()
    config.write_jsonl(config.ITEMS / "eye.jsonl", eye)
    config.write_jsonl(config.ITEMS / "photos.jsonl", photos)
    print(len(eye), "eye items,", len(photos), "photo items,", len(planned_calls(eye, photos)), "calls")


if __name__ == "__main__":
    main()
