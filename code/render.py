import io
import sys
import zipfile

from PIL import Image, ImageDraw, ImageFont

import config
import items
import zoomcheck

Image.MAX_IMAGE_PIXELS = None

QUALITY = 95
FLOOR = 85
MAX_BYTES = 5_000_000
ORIGINAL_LIMIT = 20_000_000
OVERVIEW_LONG_EDGE = 2576
RED = (230, 0, 0)
EXTENSIONS = {"JPEG": ("jpg", "image/jpeg"), "PNG": ("png", "image/png"), "WEBP": ("webp", "image/webp")}
CHANGED = []


def jpeg(image, quality=QUALITY):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality, subsampling=0)
    return buffer.getvalue()


def fitted(image):
    data = jpeg(image)
    if len(data) <= MAX_BYTES:
        return data, QUALITY
    best, low, high = None, FLOOR, QUALITY - 1
    while low <= high:
        middle = (low + high) // 2
        data = jpeg(image, middle)
        if len(data) <= MAX_BYTES:
            best, low = (data, middle), middle + 1
        else:
            high = middle - 1
    return best or (jpeg(image, FLOOR), FLOOR)


def listed():
    return {row["path"]: row for row in config.read_jsonl(config.IMAGES_LOG)}


def store(path, data, how, known):
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        kind = image.format
    row = {"path": config.relative(path), "width": width, "height": height, "bytes": len(data),
           "sha256": config.sha256_bytes(data), "mime": EXTENSIONS[kind][1], **how}
    config.write_bytes(path, data)
    previous = known.get(row["path"])
    if previous:
        if previous["sha256"] != row["sha256"]:
            CHANGED.append(row["path"])
        return previous
    config.append_jsonl(config.IMAGES_LOG, row)
    known[row["path"]] = row
    return row


def eye_path(item, old=False):
    return config.IMAGES / "eye" / f"{item['id']}{'.old' if old else ''}.jpg"


def draw_eye(item):
    canvas = Image.new("RGB", (item["width"], item["height"]), items.BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    cx, cy = item["ring"]["center"]
    radius = item["ring"]["radius"]
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=RED, width=item["ring"]["width"])
    x0, y0, x1, y1 = item["label"]
    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255))
    font = ImageFont.truetype(str(items.font_path()), item["font_size"])
    left, top, right, bottom = font.getbbox(item["code"], anchor="ls")
    pad_x = ((x1 - x0) - (right - left)) / 2
    pad_y = ((y1 - y0) - (bottom - top)) / 2
    draw.text((x0 + pad_x - left, y0 + pad_y - top), item["code"], fill=(0, 0, 0), font=font, anchor="ls")
    return canvas


def old_tier(image):
    size = zoomcheck.claude_size(*image.size, zoomcheck.RULES["anthropic:old-tier"])
    return image if size == image.size else image.resize(size, Image.LANCZOS)


def render_eye(eye, known):
    for number, item in enumerate(eye):
        if eye_path(item).exists() and eye_path(item, True).exists():
            continue
        data = jpeg(draw_eye(item))
        store(eye_path(item), data, {"kind": "eye", "item": item["id"], "quality": QUALITY}, known)
        with Image.open(io.BytesIO(data)) as native:
            small = jpeg(old_tier(native))
        store(eye_path(item, True), small, {"kind": "old-tier", "item": item["id"],
                                            "from": config.relative(eye_path(item)), "quality": QUALITY}, known)
        if number % 100 == 99:
            print(number + 1, "eye images", flush=True)


def source_bytes(item):
    if item["source"].startswith("data/downloads/mme/data.zip:"):
        member = item["source"].split(":", 1)[1]
        with zipfile.ZipFile(config.DOWNLOADS / "mme" / "data.zip") as archive:
            return archive.read(member)
    return (config.ROOT / item["source"]).read_bytes()


def decoded(item):
    with Image.open(io.BytesIO(source_bytes(item))) as image:
        image.load()
        return image.copy()


def prepare_photo(item, known):
    stem = config.IMAGES / "photos" / item["image"]
    for extension, _ in EXTENSIONS.values():
        path = stem.with_name(f"{stem.name}.{extension}")
        if path.exists() and config.relative(path) in known:
            return known[config.relative(path)]
    data = source_bytes(item)
    with Image.open(io.BytesIO(data)) as image:
        kind = image.format
        how = {"kind": "photo", "image": item["image"], "source": item["source"], "quality": None}
        extension = EXTENSIONS[kind][0] if kind in EXTENSIONS else "jpg"
        if len(data) > ORIGINAL_LIMIT or kind not in EXTENSIONS:
            how.update(quality=QUALITY, reencoded_from={"format": kind, "bytes": len(data)})
            data, extension = jpeg(image), "jpg"
    return store(stem.with_name(f"{stem.name}.{extension}"), data, how, known)


def overview_path(item):
    return config.IMAGES / "overviews" / f"{item['image']}.jpg"


def prepare_overview(item, photo, image, known):
    if max(photo["width"], photo["height"]) <= OVERVIEW_LONG_EDGE:
        return photo
    scale = OVERVIEW_LONG_EDGE / max(image.size)
    size = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    data = jpeg(image.resize(size, Image.LANCZOS))
    return store(overview_path(item), data, {"kind": "overview", "image": item["image"], "source": item["source"],
                                             "quality": QUALITY}, known)


def tile_boxes(width, height, n):
    overlap = zoomcheck.TILE_OVERLAP

    def spans(length):
        size = length / (n - overlap * (n - 1))
        starts = [round(index * (1 - overlap) * size) for index in range(n)]
        return [(start, min(length, round(start + size)) if index < n - 1 else length)
                for index, start in enumerate(starts)]

    return [(x0, y0, x1, y1) for y0, y1 in spans(height) for x0, x1 in spans(width)]


def tile_path(item, n, index):
    return config.IMAGES / "tiles" / f"{item['image']}.t{n}-{index}.jpg"


def prepare_tiles(item, image, n, known):
    for index, box in enumerate(tile_boxes(image.width, image.height, n)):
        data, quality = fitted(image.crop(box))
        store(tile_path(item, n, index), data, {"kind": "tile", "image": item["image"], "grid": n, "index": index,
                                                "box": list(box), "source": item["source"], "quality": quality},
              known)


def crop_box(width, height, box, margin=0.20, min_pixels=512, min_share=0.10):
    x0, y0, x1, y1 = (float(value) for value in box)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"box {box} is not left, top, right, bottom fractions in order")
    x0, x1, y0, y1 = x0 * width, x1 * width, y0 * height, y1 * height
    side = max(min_pixels, min_share * max(width, height))

    def span(low, high, length):
        extra = margin * (high - low)
        low, high = low - extra, high + extra
        if high - low < side:
            middle = (low + high) / 2
            low, high = middle - side / 2, middle + side / 2
        if high - low >= length:
            return 0, length
        shift = max(0 - low, 0) - max(high - length, 0)
        return round(low + shift), round(high + shift)

    left, right = span(x0, x1, width)
    top, bottom = span(y0, y1, height)
    return left, top, right, bottom


def prepare_crop(item, model, box, known):
    path = config.IMAGES / "zoom" / f"{item['id']}.{model}.jpg"
    if path.exists() and config.relative(path) in known:
        return known[config.relative(path)]
    image = decoded(item)
    pixels = crop_box(image.width, image.height, box)
    data, quality = fitted(image.crop(pixels))
    return store(path, data, {"kind": "crop", "item": item["id"], "model": model, "box": box,
                              "pixels": list(pixels), "source": item["source"], "quality": quality}, known)


def render_photos(photos, known):
    for number, item in enumerate(photos):
        photo = prepare_photo(item, known)
        grids = (2, 3) if max(item["width"], item["height"]) >= zoomcheck.TILE3_MIN_EDGE else (2,)
        done = [tile_path(item, n, index) for n in grids for index in range(n * n)]
        if max(item["width"], item["height"]) > OVERVIEW_LONG_EDGE:
            done.append(overview_path(item))
        if all(path.exists() for path in done):
            continue
        image = decoded(item)
        prepare_overview(item, photo, image, known)
        for n in grids:
            prepare_tiles(item, image, n, known)
        if number % 50 == 49:
            print(number + 1, "photos", flush=True)


def render_crops(photos, known):
    lookup = {item["id"]: item for item in photos}
    for model, spec in items.MODELS.items():
        for record in config.read_jsonl(config.CALLS / f"{model}.jsonl"):
            if record["setting"] == "zoom2":
                prepare_crop(lookup[record["item"]], model, record["box"], known)


def main(parts):
    eye, photos = items.load_items()
    known = listed()
    if "eye" in parts:
        render_eye(eye, known)
    if "photos" in parts:
        render_photos(photos, known)
    if "crops" in parts:
        render_crops(photos, known)
    if CHANGED:
        print(f"{len(CHANGED)} images differ from the ones recorded in data/images.jsonl; the recorded rows are kept")


if __name__ == "__main__":
    main(sys.argv[1:] or ["eye", "photos"])
