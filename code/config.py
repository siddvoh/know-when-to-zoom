import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOWNLOADS = DATA / "downloads"
IMAGES = DATA / "images"
ITEMS = DATA / "items"
CALLS = DATA / "calls"
RESULTS = DATA / "results"
FIGURES = ROOT / "figures"
SOURCES_PATH = DATA / "sources.json"
IMAGES_LOG = DATA / "images.jsonl"
ENV_PATH = ROOT / ".env"
SEED = 20261017
LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(items, *salt):
    prefix = ":".join(str(part) for part in (SEED,) + salt)
    return sorted(items, key=lambda item: sha256_text(f"{prefix}:{item}"))


def seed_of(*parts):
    return int(sha256_text(":".join(str(part) for part in (SEED,) + parts))[:8], 16)


def environment(name):
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set; put it in {ENV_PATH}")
    return value


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode())


def write_jsonl(path, rows):
    write_bytes(path, "".join(canonical(row) + "\n" for row in rows).encode())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, open(path, "a") as handle:
        handle.write(canonical(row) + "\n")
