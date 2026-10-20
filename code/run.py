import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import config
import items
import providers
import render

MAX_TOKENS = 8192
LONG_MAX_TOKENS = 32768
WORKERS = 8
PARTS = ("eye", "photos", "zoom")


def context():
    eye, photos = items.load_items()
    images = render.listed()
    photo_rows, overview_rows, tile_rows = {}, {}, {}
    for row in images.values():
        if row["kind"] == "photo":
            photo_rows[row["image"]] = row
        elif row["kind"] == "overview":
            overview_rows[row["image"]] = row
        elif row["kind"] == "tile":
            tile_rows[(row["image"], row["grid"], row["index"])] = row
    return {"eye": {item["id"]: item for item in eye}, "photos": {item["id"]: item for item in photos},
            "images": images, "photo_rows": photo_rows, "overview_rows": overview_rows, "tile_rows": tile_rows,
            "done": {model: finished(model) for model in items.MODELS}}


def finished(model):
    return {row["id"]: row for row in config.read_jsonl(config.CALLS / f"{model}.jsonl")}


def with_setting(row, setting):
    return dict(row, setting=setting)


def request_of(ctx, call, spec):
    setting = call["setting"]
    if call["part"] == "eye":
        item = ctx["eye"][call["item"]]
        row = ctx["images"][config.relative(render.eye_path(item, setting == "old-tier"))]
        return [with_setting(row, setting)], items.EYE_PROMPT, {}
    item = ctx["photos"][call["item"]]
    photo = ctx["photo_rows"][item["image"]]
    overview = ctx["overview_rows"].get(item["image"], photo)
    default = spec["default"]
    if setting == "as_is":
        return [with_setting(photo, default)], items.photo_prompt(item), {}
    if setting == "max":
        return [with_setting(photo, spec["max"])], items.photo_prompt(item), {}
    if setting in ("tile2", "tile3"):
        n = int(setting[-1])
        tiles = [ctx["tile_rows"][(item["image"], n, index)] for index in range(n * n)]
        return ([with_setting(row, default) for row in [overview] + tiles],
                items.photo_prompt(item, items.tile_lead(n * n)), {})
    if setting == "zoom1":
        return [with_setting(overview, default)], items.box_prompt(item), {}
    first = ctx["done"][call["model"]].get(f"{call['model']}|photos|zoom1|{item['id']}")
    if first is None:
        return None, None, None
    box, source = (box_of(first["parsed"]), "model") if first["status"] == "ok" else ([0.0, 0.0, 1.0, 1.0], "full")
    crop = render.prepare_crop(item, call["model"], box, ctx["images"])
    return ([with_setting(overview, default), with_setting(crop, default)],
            items.photo_prompt(item, items.ZOOM_LINE), {"box": box, "box_source": source, "crop": crop["path"]})


def first_object(text):
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text, start)
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return None


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def box_problem(value):
    if value is None:
        return "no JSON object"
    if any(edge not in value for edge in items.EDGES):
        return "missing edge"
    edges = [value[edge] for edge in items.EDGES]
    if not all(number(edge) for edge in edges):
        return "not a number"
    if any(edge > 1 for edge in edges):
        return "above 1"
    if any(edge < 0 for edge in edges):
        return "below 0"
    if not (value["left"] < value["right"] and value["top"] < value["bottom"]):
        return "edges out of order"
    return None


def box_of(parsed):
    return [float(parsed[edge]) for edge in items.EDGES]


def parse(call, item, text):
    value = first_object(text or "")
    if value is None:
        return None
    if call["part"] == "eye":
        ok = isinstance(value.get("code"), str) and isinstance(value.get("readable"), bool) \
            and number(value.get("confidence"))
    elif call["setting"] == "zoom1":
        ok = box_problem(value) is None
    else:
        letters = {option["letter"] for option in item["options"]} | {"X"}
        answer = value.get("answer")
        ok = isinstance(answer, str) and answer.strip().upper() in letters and number(value.get("confidence"))
    return value if ok else None


def ask(session, call, spec, images, text, item):
    seed = config.seed_of(call["id"]) % 2 ** 31
    images = providers.with_uploads(session, spec, images, text)
    max_tokens, repaired = MAX_TOKENS, False
    while True:
        status, payload, problem = providers.send(session, spec, providers.body(spec, images, text, max_tokens, seed))
        if status != 200:
            if providers.rejected(status, payload):
                return {"status": "rejected", "error": providers.error_message(payload)}, {}
            raise RuntimeError(problem or f"HTTP {status}: {providers.error_message(payload)}")
        info = providers.read(spec, payload)
        if info.get("error"):
            raise RuntimeError(json.dumps(info["error"]))
        if info["refused"]:
            return {"status": "refused", "error": None}, info
        if info["truncated"] and max_tokens == MAX_TOKENS:
            max_tokens = LONG_MAX_TOKENS
            continue
        answer = parse(call, item, info["text"]) if info["natural"] else None
        if answer is not None:
            return {"status": "ok", "parsed": answer, "error": None}, info
        if not repaired and not info["truncated"]:
            repaired = True
            continue
        return {"status": "unparsed", "error": None}, info


def run_call(ctx, sessions, call):
    spec = items.MODELS[call["model"]]
    images, text, extra = request_of(ctx, call, spec)
    if images is None:
        return "waiting"
    item = ctx["photos"].get(call["item"]) or ctx["eye"][call["item"]]
    try:
        outcome, info = ask(sessions[providers.account(spec)], call, spec, images, text, item)
    except Exception as error:
        print(f"{call['id']}: {error}", flush=True)
        return "error"
    record = {"id": call["id"], "model": call["model"], "part": call["part"], "setting": call["setting"],
              "item": call["item"], "repeat": call["repeat"],
              "images": [{"path": image["path"], "sha256": image["sha256"], "setting": image["setting"]}
                         for image in images],
              "status": outcome["status"], "finish_reason": info.get("finish"), "text": info.get("text"),
              "reasoning": info.get("reasoning"), "parsed": outcome.get("parsed"), "tokens": info.get("tokens"),
              "model_version": info.get("model_version"), "error": outcome["error"], "completed_at": config.now(),
              **extra}
    config.append_jsonl(config.CALLS / f"{call['model']}.jsonl", record)
    return outcome["status"]


def main(arguments):
    mode = arguments[0] if arguments else ""
    if mode not in PARTS:
        raise SystemExit("usage: python code/run.py eye | photos | zoom [--models a,b] [--workers n]")
    models = list(items.MODELS)
    if "--models" in arguments:
        models = arguments[arguments.index("--models") + 1].split(",")
    workers = int(arguments[arguments.index("--workers") + 1]) if "--workers" in arguments else WORKERS
    ctx = context()
    calls = [call for call in items.planned_calls(list(ctx["eye"].values()), list(ctx["photos"].values()))
             if call["part"] == mode and call["model"] in models and call["id"] not in ctx["done"][call["model"]]]
    groups = defaultdict(list)
    for call in calls:
        groups[providers.account(items.MODELS[call["model"]])].append(call)
    sessions = {name: providers.connect(name) for name in groups}
    print(len(calls), "calls to send", flush=True)
    pools = {name: ThreadPoolExecutor(workers) for name in groups}
    futures = [pools[name].submit(run_call, ctx, sessions, call) for name, group in groups.items() for call in group]
    print(dict(Counter(future.result() for future in futures)))
    for pool in pools.values():
        pool.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
