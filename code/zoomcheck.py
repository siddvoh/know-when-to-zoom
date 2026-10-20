import math

TILE_OVERLAP = 0.10
TILE3_MIN_EDGE = 4000
D_MIN = 16

RULES = {
    "anthropic:native": {"kind": "claude", "long_edge": 2576, "tokens": 4784, "patch": 28, "max_dimension": 8000},
    "anthropic:old-tier": {"kind": "claude", "long_edge": 1568, "tokens": 1568, "patch": 28, "max_dimension": 8000},
    "gemini:low": {"kind": "grid", "ceiling": 280},
    "gemini:medium": {"kind": "grid", "ceiling": 560},
    "gemini:high": {"kind": "grid", "ceiling": 1120},
    "gemini:ultra_high": {"kind": "grid", "ceiling": 2240},
    "gemini:default": {"kind": "grid", "ceiling": 1120},
    "openai:auto": {"kind": "openai", "patch": 32, "resize_patches": None, "reject_patches": 30000,
                    "multiplier": 1.2},
    "openai:high": {"kind": "openai", "patch": 32, "resize_patches": 2500, "reject_patches": 30000,
                    "multiplier": 1.2},
    "qwen:default": {"kind": "qwen", "patch": 32, "max_tokens": 16384},
}


def scaled(width, height, fits):
    if fits(width, height):
        return width, height
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if fits(max(1, round(width * middle)), max(1, round(height * middle))):
            low = middle
        else:
            high = middle
    return max(1, round(width * low)), max(1, round(height * low))


def claude_size(width, height, rule):
    patch = rule["patch"]

    def fits(w, h):
        return max(w, h) <= rule["long_edge"] and math.ceil(w / patch) * math.ceil(h / patch) <= rule["tokens"]

    return scaled(width, height, fits)


def patches(width, height, patch):
    return math.ceil(width / patch) * math.ceil(height / patch)


def openai_size(width, height, rule):
    patch, budget = rule["patch"], rule["resize_patches"]
    if budget is None or patches(width, height, patch) <= budget:
        return width, height
    scale = math.sqrt(patch * patch * budget / (width * height))
    scale *= min(math.floor(width * scale / patch) / (width * scale / patch),
                 math.floor(height * scale / patch) / (height * scale / patch))
    return round(width * scale, 6), round(height * scale, 6)


def qwen_size(width, height, rule):
    patch, limit = rule["patch"], rule["max_tokens"] * rule["patch"] ** 2
    w, h = max(patch, round(width / patch) * patch), max(patch, round(height / patch) * patch)
    if w * h > limit:
        beta = math.sqrt(width * height / limit)
        w, h = math.floor(width / beta / patch) * patch, math.floor(height / beta / patch) * patch
    return w, h


def grid(width, height, ceiling):
    return math.floor(math.sqrt(ceiling * width / height)), math.floor(math.sqrt(ceiling * height / width))


def grid_tokens(width, height, ceiling):
    columns, rows = grid(width, height, ceiling)
    return columns * rows


def received(width, height, rule):
    kind = rule["kind"]
    if kind == "claude":
        return None if max(width, height) > rule["max_dimension"] else claude_size(width, height, rule)
    if kind == "openai":
        size = openai_size(width, height, rule)
        return None if patches(*size, rule["patch"]) > rule["reject_patches"] else size
    if kind == "qwen":
        return qwen_size(width, height, rule)
    raise ValueError(f"no pixel size for rule kind {kind}")


def image_tokens(width, height, rule):
    if rule["kind"] == "grid":
        return grid_tokens(width, height, rule["ceiling"])
    size = received(width, height, rule)
    if size is None:
        return None
    count = patches(*size, rule["patch"])
    return math.ceil(count * rule["multiplier"]) if rule["kind"] == "openai" else count


def tokens_per_pixel(width, height, rule):
    if rule["kind"] == "grid":
        return grid(width, height, rule["ceiling"])[1] / height
    size = received(width, height, rule)
    return None if size is None else size[1] / height / rule["patch"]


def tile_size(width, height, n):
    return width / (n - TILE_OVERLAP * (n - 1)), height / (n - TILE_OVERLAP * (n - 1))


def readable(width, height, rule, threshold, d_min):
    factor = tokens_per_pixel(width, height, rule)
    return factor is not None and d_min * factor >= threshold


def choose(width, height, rule, threshold, d_min, max_rule=None, max_threshold=None):
    if readable(width, height, rule, threshold, d_min):
        return "as_is"
    if max_rule is not None and readable(width, height, max_rule, max_threshold, d_min):
        return "max"
    for n in (2, 3):
        if n == 3 and max(width, height) < TILE3_MIN_EDGE:
            continue
        if readable(*tile_size(width, height, n), rule, threshold, d_min):
            return f"tile{n}"
    return "zoom"
