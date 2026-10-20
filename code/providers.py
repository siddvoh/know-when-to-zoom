import base64
import json
import subprocess
import threading
import time

import requests

import config

CLAUDE = "https://api.anthropic.com/v1"
VERTEX = ("https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/"
          "{model}:generateContent")
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE = 1.0
GEMINI_LEVELS = {"low": "MEDIA_RESOLUTION_LOW", "medium": "MEDIA_RESOLUTION_MEDIUM",
                 "high": "MEDIA_RESOLUTION_HIGH", "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH"}
GEMINI_REFUSALS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY",
                   "IMAGE_PROHIBITED_CONTENT"}
CLAUDE_IMAGE_LIMIT = 7_500_000
CLAUDE_REQUEST_LIMIT = 32_000_000
RETRY_STATUSES = (408, 429, 500, 502, 503, 504, 529)
RETRIES = 5
TIMEOUT = 600
TOKEN_SECONDS = 600
TOKEN = {"value": None, "time": -float("inf")}
LOCK = threading.Lock()
UPLOADS = {}


def account(spec):
    return spec["vendor"] if spec["vendor"] in ("anthropic", "gemini") else "openrouter"


def connect(name):
    session = requests.Session()
    if name == "anthropic":
        session.headers.update({"x-api-key": config.environment("ANTHROPIC_API_KEY"),
                                "anthropic-version": "2023-06-01"})
    elif name == "openrouter":
        session.headers["Authorization"] = f"Bearer {config.environment('OPENROUTER_API_KEY')}"
    return session


def vertex_token(renew):
    with LOCK:
        if renew or time.monotonic() - TOKEN["time"] > TOKEN_SECONDS:
            reply = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True,
                                   check=True)
            TOKEN.update(value=reply.stdout.strip(), time=time.monotonic())
        return TOKEN["value"]


def endpoint(spec):
    if spec["vendor"] == "anthropic":
        return f"{CLAUDE}/messages"
    if spec["vendor"] == "gemini":
        return VERTEX.format(project=config.environment("GOOGLE_CLOUD_PROJECT"),
                             location=config.environment("VERTEX_LOCATION"), model=spec["model_id"])
    return OPENROUTER


def image_bytes(image):
    data = (config.ROOT / image["path"]).read_bytes()
    if config.sha256_bytes(data) != image["sha256"]:
        raise SystemExit(f"{image['path']} does not match its SHA-256 in data/images.jsonl")
    return data


def encoded(image):
    return base64.b64encode(image_bytes(image)).decode()


def claude_body(spec, images, text, max_tokens, seed):
    content = []
    for image in images:
        source = ({"type": "file", "file_id": image["file_id"]} if image.get("file_id")
                  else {"type": "base64", "media_type": image["mime"], "data": encoded(image)})
        content.append({"type": "image", "source": source})
    content.append({"type": "text", "text": text})
    return {"model": spec["model_id"], "max_tokens": max_tokens,
            "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "low"},
            "messages": [{"role": "user", "content": content}]}


def gemini_body(spec, images, text, max_tokens, seed):
    parts = []
    for image in images:
        part = {"inlineData": {"mimeType": image["mime"], "data": encoded(image)}}
        if image["setting"] in GEMINI_LEVELS:
            part["mediaResolution"] = {"level": GEMINI_LEVELS[image["setting"]]}
        parts.append(part)
    return {"contents": [{"role": "user", "parts": parts + [{"text": text}]}],
            "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": max_tokens, "seed": seed,
                                 "thinkingConfig": {"thinkingLevel": "low", "includeThoughts": True}}}


def openrouter_body(spec, images, text, max_tokens, seed):
    content = []
    for image in images:
        part = {"type": "image_url", "image_url": {"url": f"data:{image['mime']};base64,{encoded(image)}"}}
        if spec["vendor"] == "openai":
            part["image_url"]["detail"] = image["setting"]
        content.append(part)
    content.append({"type": "text", "text": text})
    body = {"model": spec["model_id"], "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "reasoning": {"effort": "low", "exclude": False}, "seed": seed, "usage": {"include": True},
            "provider": {"only": [spec["provider"]], "allow_fallbacks": False, "require_parameters": True,
                         "data_collection": "deny"}}
    if spec["vendor"] == "qwen":
        body["temperature"] = TEMPERATURE
    return body


def body(spec, images, text, max_tokens, seed):
    build = {"anthropic": claude_body, "gemini": gemini_body}.get(spec["vendor"], openrouter_body)
    return build(spec, images, text, max_tokens, seed)


def parsed(reply):
    try:
        return reply.json()
    except ValueError:
        return reply.text


def send(session, spec, payload):
    url, data = endpoint(spec), json.dumps(payload).encode()
    status = reply = None
    for number in range(RETRIES + 1):
        headers = {"content-type": "application/json"}
        if spec["vendor"] == "gemini":
            headers["Authorization"] = f"Bearer {vertex_token(status == 401)}"
        try:
            reply = session.post(url, data=data, headers=headers, timeout=TIMEOUT)
            status, answer, problem = reply.status_code, parsed(reply), None
        except requests.RequestException as error:
            reply, status, answer, problem = None, None, None, f"{type(error).__name__}: {error}"
        renew = spec["vendor"] == "gemini" and status == 401
        if status is not None and status not in RETRY_STATUSES and not renew:
            break
        if number < RETRIES:
            wait = 2 ** (number + 1)
            after = reply.headers.get("retry-after") if reply is not None else None
            if after and after.replace(".", "", 1).isdigit():
                wait = max(wait, float(after))
            time.sleep(min(60, wait))
    return status, answer, problem


def with_uploads(session, spec, images, text):
    if spec["vendor"] != "anthropic":
        return images
    sizes = [4 * -(-image["bytes"] // 3) for image in images]
    over = sum(sizes) + len(text.encode()) > CLAUDE_REQUEST_LIMIT
    result = []
    for image, size in zip(images, sizes):
        if over or size > CLAUDE_IMAGE_LIMIT:
            with LOCK:
                if image["sha256"] not in UPLOADS:
                    data = image_bytes(image)
                    reply = session.post(f"{CLAUDE}/files", timeout=TIMEOUT,
                                         files={"file": (image["sha256"][:16] + ".jpg", data, image["mime"])})
                    reply.raise_for_status()
                    UPLOADS[image["sha256"]] = reply.json()["id"]
            image = dict(image, file_id=UPLOADS[image["sha256"]])
        result.append(image)
    return result


def read_claude(payload):
    blocks = payload.get("content") or []
    usage = payload.get("usage") or {}
    stop = payload.get("stop_reason")
    cached = (usage.get("cache_read_input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0)
    return {"text": "".join(block.get("text", "") for block in blocks if block.get("type") == "text"),
            "reasoning": "".join(block.get("thinking", "") for block in blocks if block.get("type") == "thinking"),
            "finish": stop, "natural": stop == "end_turn", "truncated": stop == "max_tokens",
            "refused": stop == "refusal",
            "tokens": {"input": usage.get("input_tokens", 0) + cached, "image": None,
                       "output": usage.get("output_tokens", 0), "reasoning": None},
            "model_version": payload.get("model")}


def read_gemini(payload):
    candidate = (payload.get("candidates") or [{}])[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    usage = payload.get("usageMetadata") or {}
    finish = candidate.get("finishReason")
    blocked = (payload.get("promptFeedback") or {}).get("blockReason")
    thoughts = usage.get("thoughtsTokenCount", 0)
    image = sum(entry.get("tokenCount", 0) for entry in usage.get("promptTokensDetails") or []
                if entry.get("modality") == "IMAGE")
    return {"text": "".join(part["text"] for part in parts if "text" in part and not part.get("thought")),
            "reasoning": "".join(part["text"] for part in parts if "text" in part and part.get("thought")),
            "finish": finish or blocked, "natural": finish == "STOP", "truncated": finish == "MAX_TOKENS",
            "refused": bool(blocked) or finish in GEMINI_REFUSALS,
            "tokens": {"input": usage.get("promptTokenCount", 0), "image": image,
                       "output": usage.get("candidatesTokenCount", 0) + thoughts, "reasoning": thoughts},
            "model_version": payload.get("modelVersion")}


def read_openrouter(payload):
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = payload.get("usage") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    finish = choice.get("finish_reason")
    return {"text": content, "reasoning": message.get("reasoning") or "", "finish": finish,
            "natural": finish == "stop", "truncated": finish == "length", "refused": finish == "content_filter",
            "tokens": {"input": usage.get("prompt_tokens", 0),
                       "image": (usage.get("prompt_tokens_details") or {}).get("image_tokens"),
                       "output": usage.get("completion_tokens", 0),
                       "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")},
            "model_version": payload.get("model"), "error": choice.get("error") or payload.get("error")}


def read(spec, payload):
    return {"anthropic": read_claude, "gemini": read_gemini}.get(spec["vendor"], read_openrouter)(payload)


def error_message(payload):
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return str(payload)
    raw = (error.get("metadata") or {}).get("raw")
    try:
        inner = json.loads(raw)["error"]["message"]
    except (TypeError, ValueError, KeyError):
        inner = None
    return inner or error.get("message") or json.dumps(error)


def rejected(status, payload):
    words = ("image", "pixel", "patch", "dimension", "resolution")
    return status in (400, 413, 422) and any(word in error_message(payload).lower() for word in words)
