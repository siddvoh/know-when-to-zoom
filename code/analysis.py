from collections import Counter, defaultdict

import numpy as np
from scipy import stats
from scipy.special import expit

import config
import items
import zoomcheck

RESAMPLES = 2000
ALPHA = 0.05
RIDGE = 0.01
RIDGES = tuple(float(value) for value in np.logspace(-4, -1, 13))
CANNOT_TELL = "X"
CONFIDENT = 70
GAP = 0.02
TOKEN_SHARE = 0.60
AIC_GAIN = 10
SPREAD = 1.25
EIGHT_K = (7680, 4320)
PHOTOS_12MP = (4000, 3000)
SWEEP = (4, 6, 8, 11, 16, 23, 32, 45, 64)
DATASETS = ("vstar", "hrbench", "mme-rs", "mme-mon")
TABLE_RULES = ("anthropic:native", "anthropic:old-tier", "gemini:low", "gemini:medium", "gemini:high",
               "gemini:ultra_high", "openai:auto", "openai:high", "qwen:default")


def calls_of(model):
    return {row["id"]: row for row in config.read_jsonl(config.CALLS / f"{model}.jsonl")}


def rule(model, setting):
    return zoomcheck.RULES[f"{items.MODELS[model]['vendor']}:{setting}"]


def image_tokens(image, model):
    return zoomcheck.image_tokens(image["width"], image["height"], rule(model, image["setting"]))


def fit(x, y, weight=None, start=None, ridge=RIDGE):
    weight = np.ones(len(x)) if weight is None else weight
    hits, total = float(weight @ y), float(weight.sum())
    if hits == 0 or hits == total:
        return None
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.0, ridge])
    beta = np.zeros(2) if start is None else start.copy()
    for _ in range(200):
        p = expit(design @ beta)
        gradient = design.T @ (weight * (y - p)) - penalty @ beta
        hessian = (design * (weight * p * (1 - p))[:, None]).T @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        size = np.abs(step).max()
        if size > 5:
            step *= 5 / size
        beta = beta + step
        if size < 1e-10:
            break
    return beta


def log_likelihood(beta, x, y):
    z = beta[0] + beta[1] * x
    return float(np.sum(y * z - np.logaddexp(0, z)))


def c50(beta):
    if beta is None or beta[1] <= 0:
        return None
    return float(2.0 ** (-beta[0] / beta[1]))


def interval(draws):
    values = [value for value in draws if value is not None]
    if len(values) < 0.95 * len(draws):
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def resamples(strata, n, seed):
    rng = np.random.default_rng(seed)
    picks = np.empty((RESAMPLES, n), dtype=int)
    column = 0
    for members in strata:
        members = np.asarray(members)
        picks[:, column:column + len(members)] = members[rng.integers(0, len(members), (RESAMPLES, len(members)))]
        column += len(members)
    return [np.bincount(row, minlength=n).astype(float) for row in picks]


def clopper(hits, n):
    low = stats.beta.ppf(0.025, hits, n - hits + 1) if hits > 0 else 0.0
    high = stats.beta.ppf(0.975, hits + 1, n - hits) if hits < n else 1.0
    return [float(low), float(high)]


def canvas_order(name):
    return int(name.split("x")[0])


def services():
    table = {}
    for key in TABLE_RULES:
        service = zoomcheck.RULES[key]
        for name, (width, height) in (("12mp", PHOTOS_12MP), ("8k", EIGHT_K)):
            entry = {"tokens": zoomcheck.image_tokens(width, height, service)}
            if service["kind"] == "grid":
                entry["grid"] = list(zoomcheck.grid(width, height, service["ceiling"]))
            else:
                size = zoomcheck.received(width, height, service)
                entry["received"] = list(size) if size else None
                if service["kind"] == "openai":
                    entry["patches"] = zoomcheck.patches(width, height, service["patch"])
            table[f"{key}|{name}"] = entry
    return table


def token_match(calls, images):
    result = {}
    for model, spec in items.MODELS.items():
        eye_offsets, grid_off, rejection_mismatches = Counter(), Counter(), 0
        for record in calls[model].values():
            sent = [dict(images[image["path"]], setting=image["setting"]) for image in record["images"]]
            if spec["vendor"] == "openai":
                over = any(zoomcheck.received(image["width"], image["height"], rule(model, image["setting"])) is None
                           for image in sent)
                rejection_mismatches += over != (record["status"] == "rejected")
            if record["status"] != "ok":
                continue
            expected = sum(image_tokens(image, model) for image in sent)
            measured = record["tokens"]["image"] if spec["vendor"] == "gemini" else record["tokens"]["input"]
            if record["part"] == "eye":
                eye_offsets[measured - expected] += 1
            elif spec["vendor"] == "gemini":
                sides = {side for image in sent
                         for side in zoomcheck.grid(image["width"], image["height"], rule(model, image["setting"])["ceiling"])}
                difference = measured - expected
                grid_off["exact" if difference == 0 else "one row or column" if abs(difference) in sides else "other"] += 1
        result[model] = {"eye_offsets": {str(key): value for key, value in sorted(eye_offsets.items())}}
        if spec["vendor"] == "gemini":
            result[model]["photo_calls"] = dict(grid_off)
        if spec["vendor"] == "openai":
            result[model]["rejections_not_explained_by_30000_patches"] = rejection_mismatches
    return result


def eye_rows(eye, calls):
    rows = []
    for model in items.MODELS:
        for record in calls[model].values():
            if record["part"] != "eye":
                continue
            item = eye[record["item"]]
            factor = zoomcheck.tokens_per_pixel(item["width"], item["height"], rule(model, record["setting"]))
            parsed = record["parsed"] or {}
            code = str(parsed.get("code") or "").strip().upper()
            rows.append({"model": model, "setting": record["setting"], "item": item["id"],
                         "canvas": f"{item['width']}x{item['height']}", "cap": item["cap"],
                         "tokens": item["cap"] * factor if factor else None, "repeat": record["repeat"],
                         "status": record["status"], "code": code, "exact": code == item["code"],
                         "readable": parsed.get("readable") is True, "confidence": parsed.get("confidence")})
    return rows


def fit_group(group, seed):
    canvases = sorted({row["canvas"] for row in group}, key=canvas_order)
    caps = np.array([row["cap"] for row in group], float)
    x_px, x_tok = np.log2(caps), np.log2([row["tokens"] for row in group])
    y = np.array([row["exact"] for row in group], float)
    where = np.array([row["canvas"] for row in group])
    factor = {canvas: next(row["tokens"] / row["cap"] for row in group if row["canvas"] == canvas)
              for canvas in canvases}
    received, original = fit(x_tok, y), fit(x_px, y)
    masks = {canvas: where == canvas for canvas in canvases}
    per_canvas = {canvas: fit(x_px[mask], y[mask]) for canvas, mask in masks.items()}
    strata = defaultdict(list)
    for index, row in enumerate(group):
        strata[(row["canvas"], row["cap"])].append(index)
    draws = {"received": [], **{canvas: [] for canvas in canvases}}
    for weight in resamples(list(strata.values()), len(group), seed):
        draws["received"].append(c50(fit(x_tok, y, weight, received)))
        for canvas, mask in masks.items():
            draws[canvas].append(c50(fit(x_px[mask], y[mask], weight[mask], per_canvas[canvas])))
    rows = {}
    for canvas in canvases:
        mask = masks[canvas]
        value = c50(per_canvas[canvas])
        inside = value is not None and caps[mask].min() <= value <= caps[mask].max()
        rows[canvas] = {"n": int(mask.sum()), "exact": float(y[mask].mean()), "tokens_per_pixel": factor[canvas],
                        "c50_original": value, "ci_original": interval(draws[canvas]),
                        "c50_tokens": value * factor[canvas] if value is not None else None,
                        "identifiable": bool(inside and y[mask].min() == 0 and y[mask].max() == 1)}
    aic = {name: (4 - 2 * log_likelihood(beta, x, y)) if beta is not None else None
           for name, beta, x in (("received", received, x_tok), ("original", original, x_px))}
    used = [canvas for canvas in canvases if rows[canvas]["identifiable"]]
    tokens = [rows[canvas]["c50_tokens"] for canvas in used]
    spread = max(tokens) / min(tokens) if len(tokens) >= 2 else None
    gain = aic["original"] - aic["received"] if None not in aic.values() else None
    h1 = {"aic_gain": gain, "spread": spread, "canvases": used, "testable": spread is not None and gain is not None,
          "no_resize": len(set(round(value, 9) for value in factor.values())) == 1}
    h1["supported"] = bool(h1["testable"] and gain >= AIC_GAIN and spread <= SPREAD)
    summary = {"n": len(group), "canvases": rows, "h1": h1,
               "received": {"c50": c50(received), "ci": interval(draws["received"]),
                            "beta": received.tolist() if received is not None else None}}
    return summary, draws


def h1_at(group, ridge):
    caps = np.array([row["cap"] for row in group], float)
    x_px, x_tok = np.log2(caps), np.log2([row["tokens"] for row in group])
    y = np.array([row["exact"] for row in group], float)
    where = np.array([row["canvas"] for row in group])
    received, original = fit(x_tok, y, ridge=ridge), fit(x_px, y, ridge=ridge)
    tokens = []
    for canvas in sorted(set(where), key=canvas_order):
        mask = where == canvas
        value = c50(fit(x_px[mask], y[mask], ridge=ridge))
        if value is not None and caps[mask].min() <= value <= caps[mask].max() and y[mask].min() == 0 \
                and y[mask].max() == 1:
            factor = next(row["tokens"] / row["cap"] for row in group if row["canvas"] == canvas)
            tokens.append(value * factor)
    if len(tokens) < 2 or received is None or original is None:
        return None
    gain = 2 * log_likelihood(received, x_tok, y) - 2 * log_likelihood(original, x_px, y)
    return bool(gain >= AIC_GAIN and max(tokens) / min(tokens) <= SPREAD)


def ridge_sensitivity(groups):
    return {"ridges": list(RIDGES), "h1": {key: [h1_at(group, ridge) for ridge in RIDGES]
                                            for key, group in sorted(groups.items())}}


def eye_cells(rows):
    cells = defaultdict(list)
    for row in rows:
        if row["repeat"] == 1:
            cells[(row["model"], row["setting"], row["canvas"], row["cap"])].append(row)
    table = []
    for (model, setting, canvas, cap), group in sorted(cells.items(), key=lambda pair: (
            pair[0][0], pair[0][1], canvas_order(pair[0][2]), pair[0][3])):
        answered = [row for row in group if row["status"] == "ok"]
        table.append({"model": model, "setting": setting, "canvas": canvas, "cap": cap, "n": len(group),
                      "rejected": sum(row["status"] == "rejected" for row in group),
                      "tokens": answered[0]["tokens"] if answered else None,
                      "exact": float(np.mean([row["exact"] for row in answered])) if answered else None})
    return table


def claims(rows, fits):
    result = {}
    for model in items.MODELS:
        below = [row for row in rows if row["model"] == model and row["status"] == "ok" and row["repeat"] == 1
                 and fits[f"{model}|{row['setting']}"]["received"]["c50"] is not None
                 and row["tokens"] < fits[f"{model}|{row['setting']}"]["received"]["c50"] / 2]
        confident = [isinstance(row["confidence"], (int, float)) and row["confidence"] >= CONFIDENT for row in below]
        hits = sum(row["readable"] or high for row, high in zip(below, confident))
        test = stats.binomtest(hits, len(below), 0.70, alternative="greater") if below else None
        result[model] = {"n": len(below), "claims": hits, "rate": hits / len(below) if below else None,
                         "ci": clopper(hits, len(below)) if below else None,
                         "readable_true": sum(row["readable"] for row in below), "confident": sum(confident),
                         "exact": sum(row["exact"] for row in below), "p": test.pvalue if test else None}
    return result


def repeat_agreement(rows):
    first = {(row["model"], row["setting"], row["item"]): row for row in rows if row["repeat"] == 1}
    pairs = defaultdict(list)
    for row in rows:
        if row["repeat"] == 2 and row["status"] == "ok":
            other = first.get((row["model"], row["setting"], row["item"]))
            if other and other["status"] == "ok":
                pairs[f"{row['model']}|{row['setting']}"].append((row, other))
    return {key: {"pairs": len(group), "same_code": float(np.mean([a["code"] == b["code"] for a, b in group]))}
            for key, group in sorted(pairs.items())}


def tier_change(fits, draws):
    result = {}
    for model, spec in items.MODELS.items():
        if spec["vendor"] != "anthropic":
            continue
        for canvas in ("4096x3072", "8000x6000"):
            width, height = (int(side) for side in canvas.split("x"))
            native = fits[f"{model}|native"]["canvases"][canvas]["c50_original"]
            old = fits[f"{model}|old-tier"]["canvases"][canvas]["c50_original"]
            ratios = [a / b if a is not None and b is not None else None
                      for a, b in zip(draws[f"{model}|old-tier"][canvas], draws[f"{model}|native"][canvas])]
            expected = (zoomcheck.claude_size(width, height, zoomcheck.RULES["anthropic:native"])[0]
                        / zoomcheck.claude_size(width, height, zoomcheck.RULES["anthropic:old-tier"])[0])
            result[f"{model}|{canvas}"] = {"native": native, "old_tier": old, "ratio": old / native,
                                           "ci": interval(ratios), "expected": expected}
    return result


def gemini_resolution(fits, draws):
    result = {}
    for model, spec in items.MODELS.items():
        if spec["vendor"] != "gemini":
            continue
        reference = f"{model}|ultra_high"
        base = fits[reference]["canvases"]["1024x768"]["c50_original"]
        for setting in spec["eye"]:
            key = f"{model}|{setting}"
            for canvas, entry in fits[key]["canvases"].items():
                edge = canvas_order(canvas)
                ratios = [edge * a / b if a and b else None
                          for a, b in zip(draws[reference]["1024x768"], draws[key][canvas])]
                long_edge = edge * base / entry["c50_original"]
                columns = zoomcheck.grid(4, 3, rule(model, setting)["ceiling"])[0]
                result[f"{key}|{canvas}"] = {"effective_long_edge": long_edge, "ci": interval(ratios),
                                             "pixels_per_grid_cell": long_edge / columns}
    return result


def eight_k(fits):
    width, height = EIGHT_K
    rows = []
    for model, spec in items.MODELS.items():
        for setting in spec["eye"]:
            service = rule(model, setting)
            fitted = fits[f"{model}|{setting}"]
            factor = zoomcheck.tokens_per_pixel(width, height, service)
            if service["kind"] == "grid":
                readable = fitted["canvases"]["8000x6000"]["c50_original"] * width / 8000
            else:
                readable = fitted["received"]["c50"] / factor if factor else None
            rows.append({"model": model, "setting": setting, "tokens": zoomcheck.image_tokens(width, height, service),
                         "rejected": factor is None, "readable_cap": readable})
    return rows


def photo_outcomes(photos, images, calls):
    outcomes = {}
    for model in items.MODELS:
        actions = defaultdict(dict)
        for record in calls[model].values():
            if record["part"] == "eye":
                continue
            parsed = record["parsed"] or {}
            answer, confidence = str(parsed.get("answer") or "").strip().upper() or None, parsed.get("confidence")
            answered = record["status"] == "ok"
            actions[record["setting"]][record["item"]] = {
                "correct": answered and answer == photos[record["item"]]["answer"],
                "cannot_tell": answered and answer == CANNOT_TELL,
                "confident": answered and answer not in (None, CANNOT_TELL)
                and isinstance(confidence, (int, float)) and confidence >= CONFIDENT,
                "status": record["status"],
                "tokens": 0 if record["status"] == "rejected" else sum(
                    image_tokens(dict(images[image["path"]], setting=image["setting"]), model)
                    for image in record["images"])}
        if "zoom2" in actions:
            actions["zoom"] = {key: dict(step, tokens=step["tokens"] + actions["zoom1"][key]["tokens"])
                               for key, step in actions.pop("zoom2").items()}
        actions.pop("zoom1", None)
        outcomes[model] = dict(actions)
    return outcomes


def decide(model, item, limits, d_min):
    spec = items.MODELS[model]
    default, top = spec["default"], spec.get("max")
    eye_default = "high" if spec["vendor"] == "gemini" and default == "default" else default
    return zoomcheck.choose(item["width"], item["height"], rule(model, default), limits[(model, eye_default)], d_min,
                            rule(model, top) if top else None, limits[(model, top)] if top else None)


def strategy_rows(model, outcomes, photo_list, limits, d_min):
    actions = outcomes[model]
    rows = {name: [] for name in ("as_is", "max", "tile2", "tile3", "zoom", "check", "oracle")}
    for item in photo_list:
        available = {name: actions[name][item["id"]] for name in items.MODELS[model]["actions"]
                     if name in actions and item["id"] in actions[name]}
        rows["as_is"].append(available["as_is"])
        if "max" in available:
            rows["max"].append(available["max"])
        if "tile2" in available:
            rows["tile2"].append(available["tile2"])
            rows["tile3"].append(available.get("tile3", available["tile2"]))
            rows["zoom"].append(available["zoom"])
            rows["check"].append(available[decide(model, item, limits, d_min)])
            good = [step for step in available.values() if step["correct"]]
            rows["oracle"].append(min(good or available.values(), key=lambda step: step["tokens"]))
    return {name: steps for name, steps in rows.items() if steps}


def summarize(steps, seed):
    correct = np.array([step["correct"] for step in steps], float)
    rng = np.random.default_rng(seed)
    draws = correct[rng.integers(0, len(steps), (RESAMPLES, len(steps)))].mean(axis=1)
    return {"n": len(steps), "accuracy": float(correct.mean()),
            "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "tokens": float(np.mean([step["tokens"] for step in steps])),
            "rejected": sum(step["status"] == "rejected" for step in steps)}


def noninferior(check, baseline, seed):
    a = np.array([step["correct"] for step in check], float)
    b = np.array([step["correct"] for step in baseline], float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(a), (RESAMPLES, len(a)))
    differences = a[picks].mean(axis=1) - b[picks].mean(axis=1)
    share = sum(step["tokens"] for step in check) / sum(step["tokens"] for step in baseline)
    low = float(np.percentile(differences, 2.5))
    return {"difference": float(a.mean() - b.mean()), "ci": [low, float(np.percentile(differences, 97.5))],
            "token_share": share, "p": float(np.mean(differences < -GAP)),
            "rule": bool(low >= -GAP and share <= TOKEN_SHARE)}


def photos_results(photo_list, outcomes, limits):
    strategies, datasets, sweep, h3 = {}, [], [], {}
    for model, spec in items.MODELS.items():
        if "tile2" not in spec["actions"]:
            strategies[model] = {"as_is": summarize([outcomes[model]["as_is"][item["id"]] for item in photo_list],
                                                    config.seed_of("strategy", model, "as_is"))}
            continue
        rows = strategy_rows(model, outcomes, photo_list, limits, zoomcheck.D_MIN)
        strategies[model] = {name: summarize(steps, config.seed_of("strategy", model, name))
                             for name, steps in rows.items()}
        strategies[model]["check"]["choices"] = dict(Counter(
            decide(model, item, limits, zoomcheck.D_MIN) for item in photo_list))
        for name, steps in rows.items():
            for dataset in DATASETS:
                chosen = [step for step, item in zip(steps, photo_list) if item["dataset"] == dataset]
                datasets.append({"model": model, "strategy": name, "dataset": dataset, "n": len(chosen),
                                 "accuracy": float(np.mean([step["correct"] for step in chosen])),
                                 "rejected": sum(step["status"] == "rejected" for step in chosen)})
        as_is, tile3 = strategies[model]["as_is"], strategies[model]["tile3"]
        for d_min in SWEEP:
            steps = strategy_rows(model, outcomes, photo_list, limits, d_min)["check"]
            accuracy = float(np.mean([step["correct"] for step in steps]))
            tokens = float(np.mean([step["tokens"] for step in steps]))
            sweep.append({"model": model, "d_min": d_min, "accuracy": accuracy, "tokens": tokens,
                          "share_of_tile_gain": (accuracy - as_is["accuracy"]) / (tile3["accuracy"] - as_is["accuracy"]),
                          "share_of_tile_tokens": tokens / tile3["tokens"],
                          "choices": dict(Counter(decide(model, item, limits, d_min) for item in photo_list))})
        h3[model] = noninferior(rows["check"], rows["tile3"], config.seed_of("h3", model))
    return strategies, datasets, sweep, h3


def flag_test(photo_list, outcomes, limits):
    result = {}
    for model in items.MODELS:
        pairs = [(decide(model, item, limits, zoomcheck.D_MIN) != "as_is", outcomes[model]["as_is"][item["id"]])
                 for item in photo_list]
        rejected = [flag for flag, step in pairs if step["status"] == "rejected"]
        pairs = [(flag, step) for flag, step in pairs if step["status"] != "rejected"]
        flagged = np.array([flag for flag, _ in pairs])
        correct = np.array([step["correct"] for _, step in pairs], float)
        entry = {"flagged": int(flagged.sum()), "unflagged": int((~flagged).sum()),
                 "rejected_excluded": len(rejected), "rejected_flagged": sum(rejected)}
        if flagged.all() or not flagged.any():
            result[model] = dict(entry, testable=False, p=None, rule=False)
            continue
        rng = np.random.default_rng(config.seed_of("h4", model))
        differences = []
        for _ in range(RESAMPLES):
            pick = rng.integers(0, len(pairs), len(pairs))
            chosen_flag, chosen = flagged[pick], correct[pick]
            if chosen_flag.all() or not chosen_flag.any():
                continue
            differences.append(chosen[~chosen_flag].mean() - chosen[chosen_flag].mean())
        low = float(np.percentile(differences, 2.5))
        result[model] = dict(entry, testable=True, difference=float(correct[~flagged].mean() - correct[flagged].mean()),
                             ci=[low, float(np.percentile(differences, 97.5))],
                             p=float(np.mean(np.array(differences) <= 0)), rule=low > 0)
    return result


def holm(tests):
    ordered = sorted((test for test in tests if test["p"] is not None), key=lambda test: test["p"])
    running = 0.0
    for rank, test in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * test["p"]))
        test["p_holm"] = running
        test["supported"] = bool(test["rule"] and running <= ALPHA)
    for test in tests:
        if test["p"] is None:
            test.update(p_holm=None, supported=False)
    return tests


def hypotheses(fits, h2, h3, h4):
    rows = [{"id": "H1", "model": key.split("|")[0], "setting": key.split("|")[1], **fit["h1"]}
            for key, fit in fits.items()]
    family = [{"id": "H2", "model": model, "p": entry["p"], "rate": entry["rate"],
               "rule": bool(entry["ci"] and entry["ci"][0] >= 0.70)} for model, entry in h2.items()]
    family += [{"id": "H3", "model": model, "p": entry["p"], "rule": entry["rule"]} for model, entry in h3.items()]
    family += [{"id": "H4", "model": model, "p": entry["p"], "rule": entry["rule"]} for model, entry in h4.items()]
    return rows + holm(family)


def lost_details(photo_list, outcomes):
    result = {}
    for model, spec in items.MODELS.items():
        if "tile2" not in spec["actions"]:
            continue
        tiles = {**outcomes[model]["tile2"], **outcomes[model]["tile3"]}
        lost = [outcomes[model]["as_is"][item["id"]] for item in photo_list
                if tiles[item["id"]]["correct"] and not outcomes[model]["as_is"][item["id"]]["correct"]]
        result[model] = {"lost": len(lost), "rejected": sum(step["status"] == "rejected" for step in lost),
                         "cannot_tell": sum(step["cannot_tell"] for step in lost),
                         "confident_wrong": sum(step["confident"] for step in lost),
                         "other_wrong": sum(step["status"] == "ok" and not step["cannot_tell"] and not step["confident"]
                                            for step in lost),
                         "unparsed": sum(step["status"] == "unparsed" for step in lost),
                         "flipped": sum(outcomes[model]["as_is"][item["id"]]["correct"] and not tiles[item["id"]]["correct"]
                                        for item in photo_list)}
    return result


def shrink_split(photo_list, outcomes):
    result = {}
    for model, spec in items.MODELS.items():
        if spec["vendor"] != "anthropic":
            continue
        service = rule(model, spec["default"])
        tiles = {**outcomes[model]["tile2"], **outcomes[model]["tile3"]}
        for kept in (True, False):
            chosen = [item for item in photo_list
                      if (zoomcheck.received(item["width"], item["height"], service)
                          == (item["width"], item["height"])) == kept]
            result[f"{model}|{'full size' if kept else 'shrunk'}"] = {
                "items": len(chosen),
                "as_is": float(np.mean([outcomes[model]["as_is"][item["id"]]["correct"] for item in chosen])),
                "tile3": float(np.mean([tiles[item["id"]]["correct"] for item in chosen]))}
    return result


def shrink_rule(photo_list, outcomes):
    result = {}
    for model, spec in items.MODELS.items():
        if spec["vendor"] != "anthropic":
            continue
        service = rule(model, spec["default"])
        tiles = {**outcomes[model]["tile2"], **outcomes[model]["tile3"]}
        shrunk = [zoomcheck.received(item["width"], item["height"], service) != (item["width"], item["height"])
                  for item in photo_list]
        steps = [tiles[item["id"]] if tile else outcomes[model]["as_is"][item["id"]]
                 for item, tile in zip(photo_list, shrunk)]
        result[model] = {"tiled": sum(shrunk), "accuracy": float(np.mean([step["correct"] for step in steps])),
                         "token_share": sum(step["tokens"] for step in steps)
                         / sum(tiles[item["id"]]["tokens"] for item in photo_list)}
    return result


def reply_signal(photo_list, outcomes):
    result = {}
    for model, spec in items.MODELS.items():
        if "tile2" not in spec["actions"]:
            continue
        tiles = {**outcomes[model]["tile2"], **outcomes[model]["tile3"]}
        as_is = [outcomes[model]["as_is"][item["id"]] for item in photo_list]
        lost = [step for step, item in zip(as_is, photo_list) if tiles[item["id"]]["correct"] and not step["correct"]]
        right = [step for step in as_is if step["correct"]]
        result[model] = {"lost": len(lost), "lost_flagged": sum(step["status"] == "ok" and not step["confident"] for step in lost),
                         "correct": len(right), "correct_flagged": sum(not step["confident"] for step in right)}
    return result


def zoom_against_tiles(photo_list, outcomes):
    result = {}
    for model, spec in items.MODELS.items():
        if "zoom" not in spec["actions"]:
            continue
        tiles = {**outcomes[model]["tile2"], **outcomes[model]["tile3"]}
        difference = np.array([float(outcomes[model]["zoom"][item["id"]]["correct"]) - float(tiles[item["id"]]["correct"])
                               for item in photo_list])
        rng = np.random.default_rng(config.seed_of("paired", model))
        draws = difference[rng.integers(0, len(difference), (RESAMPLES, len(difference)))].mean(axis=1)
        result[model] = {"difference": float(difference.mean()),
                         "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
    return result


def reply_policy(photo_list, outcomes):
    result = {}
    for model, spec in items.MODELS.items():
        if "zoom" not in spec["actions"]:
            continue
        tiles = {**outcomes[model]["tile2"], **outcomes[model]["tile3"]}
        for fallback, second in (("tile3", tiles), ("zoom", outcomes[model]["zoom"])):
            correct, tokens, unsure = [], [], []
            for item in photo_list:
                first = outcomes[model]["as_is"][item["id"]]
                step = first if first["confident"] else second[item["id"]]
                correct.append(step["correct"])
                tokens.append(first["tokens"] + (0 if first["confident"] else step["tokens"]))
                unsure.append(not first["confident"])
            result[f"{model}|{fallback}"] = {
                "accuracy": float(np.mean(correct)), "fallback_share": float(np.mean(unsure)),
                "token_share": sum(tokens) / sum(tiles[item["id"]]["tokens"] for item in photo_list)}
    return result


def zoom_boxes(calls):
    result = {}
    for model, spec in items.MODELS.items():
        if "zoom" not in spec["actions"]:
            continue
        second = [record for record in calls[model].values() if record["setting"] == "zoom2"]
        used = sum(record["box_source"] == "model" for record in second)
        result[model] = {"questions": len(second), "model_box": used, "whole_photo": len(second) - used,
                         "share_model_box": used / len(second)}
    return result


def accepted_as_is(photo_list, outcomes):
    result = {}
    for model in items.MODELS:
        steps = [outcomes[model]["as_is"][item["id"]] for item in photo_list]
        kept = [step for step in steps if step["status"] != "rejected"]
        if len(kept) < len(steps):
            result[model] = {"photos": len(kept), "accuracy": float(np.mean([step["correct"] for step in kept]))}
    return result


def image_files(images):
    rows = images.values()
    return {"photos_reencoded": sum(row["kind"] == "photo" and "reencoded_from" in row for row in rows),
            "lowered_quality": sum(row["kind"] in ("tile", "crop") and row["quality"] < 95 for row in rows),
            "still_over_5mb": sum(row["kind"] in ("tile", "crop") and row["bytes"] > 5_000_000 for row in rows)}


def target_sizes(photo_list):
    sides = np.array([min(target["box"][2], target["box"][3]) for item in photo_list if item["dataset"] == "vstar"
                      for target in item["targets"]], float)
    return {"targets": len(sides), "median_short_side": float(np.median(sides))}


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), 6)
    return value


def build():
    eye_list, photo_list = items.load_items()
    eye = {item["id"]: item for item in eye_list}
    photos = {item["id"]: item for item in photo_list}
    images = {row["path"]: row for row in config.read_jsonl(config.IMAGES_LOG)}
    calls = {model: calls_of(model) for model in items.MODELS}

    rows = eye_rows(eye, calls)
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "ok" and row["repeat"] == 1:
            groups[f"{row['model']}|{row['setting']}"].append(row)
    fits, draws = {}, {}
    for key in sorted(groups):
        model, setting = key.split("|")
        fits[key], draws[key] = fit_group(groups[key], config.seed_of("eye", model, setting))
    h2 = claims(rows, fits)
    limits = {(model, setting): fits[f"{model}|{setting}"]["received"]["c50"]
              for model, spec in items.MODELS.items() for setting in spec["eye"]}

    outcomes = photo_outcomes(photos, images, calls)
    strategies, datasets, sweep, h3 = photos_results(photo_list, outcomes, limits)
    h4 = flag_test(photo_list, outcomes, limits)

    return clean({
        "services": services(),
        "token_match": token_match(calls, images),
        "eye": {"fits": fits, "cells": eye_cells(rows), "repeats": repeat_agreement(rows),
                "tier_change": tier_change(fits, draws), "gemini_resolution": gemini_resolution(fits, draws),
                "eight_k": eight_k(fits), "h2": h2, "ridge_sensitivity": ridge_sensitivity(groups)},
        "photos": {"strategies": strategies, "datasets": datasets, "shrink_split": shrink_split(photo_list, outcomes),
                   "shrink_rule": shrink_rule(photo_list, outcomes), "reply_signal": reply_signal(photo_list, outcomes),
                   "zoom_boxes": zoom_boxes(calls), "zoom_against_tiles": zoom_against_tiles(photo_list, outcomes),
                   "reply_policy": reply_policy(photo_list, outcomes),
                   "accepted_as_is": accepted_as_is(photo_list, outcomes), "image_files": image_files(images),
                   "lost": lost_details(photo_list, outcomes), "sweep": sweep, "h3": h3, "h4": h4,
                   "vstar_targets": target_sizes(photo_list)},
        "hypotheses": hypotheses(fits, h2, h3, h4),
    })


def main():
    config.write_json(config.RESULTS / "values.json", build())
    print(config.RESULTS / "values.json")


if __name__ == "__main__":
    main()
