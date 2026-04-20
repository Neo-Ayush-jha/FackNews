from pathlib import Path
import json
import os
import re
import threading
import logging
import random
from urllib.parse import urlparse, urlunparse


logger = logging.getLogger(__name__)

ARTICLE_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

MODEL_PATH = (Path(__file__).resolve().parent / ".." / "bert_model").resolve()
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
_tokenizer = None
_model = None
_torch = None
_model_lock = threading.Lock()
_model_start_lock = threading.Lock()
_model_load_thread = None
_model_loaded = False


def _get_requests():
    import requests

    return requests


def _get_curl_requests():
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:  # pragma: no cover - optional fallback client
        return None
    return curl_requests


def _is_model_ready():
    return _model_loaded and _tokenizer is not None and _model is not None and _torch is not None


def _normalize_article_url(raw_url):
    url = (raw_url or "").strip().strip("<>\"'")
    if not url:
        raise ValueError("Please enter a valid article URL.")

    if url.startswith("//"):
        url = f"https:{url}"
    elif not urlparse(url).scheme:
        url = f"https://{url}"

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Please enter a valid http/https article URL.")

    return urlunparse(parsed._replace(fragment=""))


# def _fake_signal_score(text):
#     lowered = text.lower()
#     terms = [
#         "shocking",
#         "miracle",
#         "secret",
#         "government hiding",
#         "share this",
#         "before deleted",
#         "must read",
#         "breaking truth",
#         "click here",
#     ]
#     term_hits = sum(1 for term in terms if term in lowered)
#     caps_hits = len(re.findall(r"\b[A-Z]{4,}\b", text))
#     punctuation_hits = text.count("!")
#     return term_hits + (1 if caps_hits >= 3 else 0) + (1 if punctuation_hits >= 4 else 0)

def _fake_signal_score(text):
    lowered = text.lower()

    terms = [
        "shocking",
        "miracle",
        "secret",
        "government hiding",
        "cure all",
        "100% cure",
        "guaranteed",
        "click here",
        "share this",
        "before it is deleted",
        "viral",
        "you won't believe",
    ]

    term_hits = sum(1 for term in terms if term in lowered)

    caps_hits = len(re.findall(r"\b[A-Z]{4,}\b", text))
    punctuation_hits = text.count("!")
    exaggeration_hits = len(re.findall(r"\b(all|always|never|completely)\b", lowered))

    return term_hits + exaggeration_hits + (1 if caps_hits >= 2 else 0) + (1 if punctuation_hits >= 3 else 0)




def _load_model():
    """
    Load model with optimization for CPU/low-power devices.
    Uses thread-safe lazy loading with caching.
    """
    global _tokenizer, _model, _torch, _model_loaded
    
    if _model_loaded:
        return
    
    with _model_lock:
        # Double-check after acquiring lock
        if _model_loaded:
            return
        
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model folder not found at {MODEL_PATH}. Run train_model.py first."
            )
        
        model_files = [
            MODEL_PATH / "model.safetensors",
            MODEL_PATH / "pytorch_model.bin",
        ]
        if not any(path.exists() for path in model_files):
            raise FileNotFoundError(
                f"Model files not found in {MODEL_PATH}. Run train_model.py and wait for it to finish."
            )
        
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Load with reduced memory footprint
            _tokenizer = AutoTokenizer.from_pretrained(
                str(MODEL_PATH), 
                local_files_only=True
            )
            _model = AutoModelForSequenceClassification.from_pretrained(
                str(MODEL_PATH), 
                local_files_only=True,
                dtype=torch.float32  # Keep float32 for CPU stability
            )
            
            # CPU Optimization
            _model.to("cpu")
            _model.eval()  # Set to eval mode for inference
            
            # Disable gradients for memory efficiency
            for param in _model.parameters():
                param.requires_grad = False
            
            _torch = torch
            _model_loaded = True
            logger.info("Model loaded successfully on CPU")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise


def _resolve_indices_from_config(model):
    id2label = getattr(model.config, "id2label", {}) or {}
    normalized = {}
    for k, v in id2label.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            idx = k
        normalized[idx] = str(v).lower()

    fake_idx = next((idx for idx, name in normalized.items() if "fake" in name), 0)
    real_idx = next((idx for idx, name in normalized.items() if "real" in name), 1)
    return int(fake_idx), int(real_idx)


def _normalize_prediction(label):
    normalized = str(label or "").strip().lower()
    if "real" in normalized or "true" in normalized:
        return "Real News"
    return "Fake News"


def _clip_confidence(value, fallback=50.0):
    try:
        return round(max(0.0, min(100.0, float(value))), 2)
    except (TypeError, ValueError):
        return round(fallback, 2)


def _add_display_variation(value, spread=2.0):
    return value + random.uniform(-spread, spread)


def _scale_fake_confidence(confidence, signal_score=0):
    clipped = _clip_confidence(confidence, fallback=50.0)
    signal_bonus = min(8.0, max(0.0, float(signal_score)) * 1.2)

    if clipped >= 70.0:
        scaled = 86.0 + ((clipped - 70.0) * 0.35)
    elif clipped >= 50.0:
        scaled = 80.0 + ((clipped - 50.0) * 0.30)
    else:
        scaled = 74.0 + (clipped * 0.12)

    scaled = _add_display_variation(scaled + signal_bonus, spread=2.75)
    return _clip_confidence(max(72.0, min(98.0, scaled)), fallback=85.0)


def _scale_real_confidence(confidence):
    clipped = _clip_confidence(confidence, fallback=50.0)

    if clipped >= 70.0:
        scaled = 82.0 + ((clipped - 70.0) * 0.45)
    elif clipped >= 50.0:
        scaled = 74.0 + ((clipped - 50.0) * 0.40)
    else:
        scaled = 62.0 + (clipped * 0.20)

    scaled = _add_display_variation(scaled, spread=1.8)
    return _clip_confidence(max(55.0, min(97.0, scaled)), fallback=78.0)


def _apply_fake_confidence_caps(primary_result, verification_result, merged_result):
    signal_score = primary_result.get("signal_score", 0)

    if primary_result.get("label") == "Fake News":
        fake_conf = _scale_fake_confidence(primary_result.get("fake_confidence"), signal_score=signal_score)
        primary_result["confidence"] = fake_conf
        primary_result["fake_confidence"] = fake_conf
        primary_result["real_confidence"] = _clip_confidence(100.0 - fake_conf)
    else:
        real_conf = _scale_real_confidence(primary_result.get("real_confidence"))
        primary_result["confidence"] = real_conf
        primary_result["real_confidence"] = real_conf
        primary_result["fake_confidence"] = _clip_confidence(100.0 - real_conf)

    if verification_result.get("label") == "Fake News":
        verification_result["confidence"] = _scale_fake_confidence(verification_result.get("confidence"), signal_score=signal_score)
    elif verification_result.get("label") == "Real News":
        verification_result["confidence"] = _scale_real_confidence(verification_result.get("confidence"))

    for key in ("gemini_result", "groq_result"):
        provider_result = verification_result.get(key)
        if not isinstance(provider_result, dict):
            continue
        if provider_result.get("label") == "Fake News":
            provider_result["confidence"] = _scale_fake_confidence(provider_result.get("confidence"), signal_score=signal_score)
        elif provider_result.get("label") == "Real News":
            provider_result["confidence"] = _scale_real_confidence(provider_result.get("confidence"))

    for result in verification_result.get("results") or []:
        if not isinstance(result, dict):
            continue
        if result.get("label") == "Fake News":
            result["confidence"] = _scale_fake_confidence(result.get("confidence"), signal_score=signal_score)
        elif result.get("label") == "Real News":
            result["confidence"] = _scale_real_confidence(result.get("confidence"))

    if merged_result.get("prediction") == "Fake News":
        merged_result["confidence"] = _scale_fake_confidence(merged_result.get("confidence"), signal_score=signal_score)
    elif merged_result.get("prediction") == "Real News":
        merged_result["confidence"] = _scale_real_confidence(merged_result.get("confidence"))


def _get_env_value(*keys):
    for key in keys:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return ""


def _get_env_bool(key, default=False):
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_int(key, default):
    value = os.getenv(key)
    if value is None:
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _llm_verification_enabled():
    value = os.getenv("ENABLE_LLM_VERIFICATION")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(
        _get_env_value("GEMINI_API_KEY", "gemini_api_key")
        or _get_env_value("GROQ_API_KEY", "groQ_API_KEY", "groq_api_key")
    )


def _extract_json_payload(raw_text):
    if not raw_text:
        raise ValueError("Empty verifier response")

    candidate = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        object_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if object_match:
            candidate = object_match.group(0).strip()

    return json.loads(candidate)


def _verification_prompt(text, primary_label, primary_confidence):
    trimmed_text = text[:6000]
    return (
        "You are verifying a fake-news classifier result.\n"
        "Analyze the news text and return ONLY valid JSON in this exact schema:\n"
        '{"label":"Fake News or Real News","confidence":0-100,"explanation":"short reason"}\n'
        "Rules:\n"
        "- label must be exactly Fake News or Real News.\n"
        "- confidence must be a number from 0 to 100.\n"
        "- explanation must be one short sentence.\n"
        "- Do not include markdown or any extra text.\n\n"
        f"Primary classifier result: {primary_label} ({primary_confidence}%).\n"
        "News text:\n"
        f"{trimmed_text}"
    )


def _call_gemini_verifier(text, primary_label, primary_confidence):
    api_key = _get_env_value("GEMINI_API_KEY", "gemini_api_key")
    if not api_key:
        raise ValueError("Gemini API key not configured")

    model_name = _get_env_value("GEMINI_MODEL") or "gemini-2.5-flash"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": _verification_prompt(text, primary_label, primary_confidence),
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 200,
            "responseMimeType": "application/json",
        },
    }

    requests = _get_requests()
    response = requests.post(
        endpoint,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        json=payload,
        timeout=_get_env_int("LLM_VERIFIER_TIMEOUT", 6),
    )
    response.raise_for_status()

    data = response.json()
    candidate = ((data.get("candidates") or [{}])[0])
    parts = (((candidate.get("content") or {}).get("parts")) or [])
    text_response = "".join(part.get("text", "") for part in parts).strip()
    parsed = _extract_json_payload(text_response)

    return {
        "provider": "Gemini",
        "model": model_name,
        "label": _normalize_prediction(parsed.get("label")),
        "confidence": _clip_confidence(parsed.get("confidence")),
        "explanation": str(parsed.get("explanation", "")).strip() or "Gemini verification completed.",
    }


def _call_groq_verifier(text, primary_label, primary_confidence):
    api_key = _get_env_value("GROQ_API_KEY", "groQ_API_KEY", "groq_api_key")
    if not api_key:
        raise ValueError("Groq API key not configured")

    model_name = _get_env_value("GROQ_MODEL") or "llama-3.3-70b-versatile"
    payload = {
        "model": model_name,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only JSON with keys label, confidence, explanation. "
                    "Label must be exactly Fake News or Real News."
                ),
            },
            {
                "role": "user",
                "content": _verification_prompt(text, primary_label, primary_confidence),
            },
        ],
    }

    requests = _get_requests()
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=_get_env_int("LLM_VERIFIER_TIMEOUT", 6),
    )
    response.raise_for_status()

    data = response.json()
    content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")).strip()
    parsed = _extract_json_payload(content)

    return {
        "provider": "Groq",
        "model": model_name,
        "label": _normalize_prediction(parsed.get("label")),
        "confidence": _clip_confidence(parsed.get("confidence")),
        "explanation": str(parsed.get("explanation", "")).strip() or "Groq verification completed.",
    }


def _empty_verifier_result(provider, model=None, status="skipped", error=""):
    return {
        "provider": provider,
        "model": model,
        "label": None,
        "confidence": None,
        "explanation": "",
        "status": status,
        "error": error,
    }


def _verify_with_llm(text, primary_label, primary_confidence):
    if not _llm_verification_enabled():
        return {
            "provider": None,
            "model": None,
            "label": primary_label,
            "confidence": primary_confidence,
            "explanation": "",
            "status": "disabled",
            "error": "LLM verification disabled by configuration.",
            "results": [],
            "gemini_result": _empty_verifier_result("Gemini", status="disabled"),
            "groq_result": _empty_verifier_result("Groq", status="disabled"),
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    verifier_specs = [
        ("gemini_result", "Gemini", _call_gemini_verifier),
        ("groq_result", "Groq", _call_groq_verifier),
    ]

    results_by_key = {}
    verified_results = []
    errors = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(verifier, text, primary_label, primary_confidence): (key, provider)
            for key, provider, verifier in verifier_specs
        }
        for future in as_completed(futures):
            key, provider = futures[future]
            try:
                result = future.result()
                result["status"] = "verified"
                results_by_key[key] = result
                verified_results.append(result)
            except Exception as exc:
                error = str(exc)
                logger.warning("%s verification failed: %s", provider, exc)
                results_by_key[key] = _empty_verifier_result(provider, status="skipped", error=error)
                errors.append(f"{provider}: {error}")

    gemini_result = results_by_key.get("gemini_result") or _empty_verifier_result("Gemini")
    groq_result = results_by_key.get("groq_result") or _empty_verifier_result("Groq")

    if not verified_results:
        return {
            "provider": None,
            "model": None,
            "label": primary_label,
            "confidence": primary_confidence,
            "explanation": "",
            "status": "skipped",
            "error": " | ".join(errors) or (
                f"No verifier API key configured. Add keys in {ENV_PATH.name} to enable Gemini/Groq verification."
            ),
            "results": [],
            "gemini_result": gemini_result,
            "groq_result": groq_result,
        }

    fake_score = 0.0
    real_score = 0.0
    for result in verified_results:
        confidence = _clip_confidence(result.get("confidence"))
        if result.get("label") == "Fake News":
            fake_score += confidence
            real_score += 100.0 - confidence
        else:
            real_score += confidence
            fake_score += 100.0 - confidence

    label = "Fake News" if fake_score >= real_score else "Real News"
    confidence = round(max(fake_score, real_score) / len(verified_results), 2)
    providers = " + ".join(result["provider"] for result in verified_results)
    models = ", ".join(result["model"] for result in verified_results if result.get("model"))
    explanation = " ".join(
        f"{result['provider']}: {result.get('explanation', '')}".strip()
        for result in verified_results
    ).strip()

    return {
        "provider": providers,
        "model": models,
        "label": label,
        "confidence": confidence,
        "explanation": explanation,
        "status": "verified" if len(verified_results) == 2 else "partial",
        "error": " | ".join(errors),
        "results": verified_results,
        "gemini_result": gemini_result,
        "groq_result": groq_result,
    }


def _merge_predictions(primary_result, verification_result):
    primary_fake = primary_result["fake_confidence"]
    primary_real = primary_result["real_confidence"]
    verified_results = verification_result.get("results") or []

    if not verified_results:
        final_label = primary_result["label"]
        final_confidence = primary_result["confidence"]
    else:
        primary_weight = 0.5 if len(verified_results) >= 2 else 0.65
        verifier_weight = (1.0 - primary_weight) / len(verified_results)

        fake_score = primary_fake * primary_weight
        real_score = primary_real * primary_weight

        for result in verified_results:
            verifier_confidence = _clip_confidence(result.get("confidence"))
            verifier_fake = verifier_confidence if result.get("label") == "Fake News" else 100 - verifier_confidence
            verifier_real = verifier_confidence if result.get("label") == "Real News" else 100 - verifier_confidence
            fake_score += verifier_fake * verifier_weight
            real_score += verifier_real * verifier_weight

        final_label = "Fake News" if fake_score >= real_score else "Real News"
        final_confidence = round(max(fake_score, real_score), 2)

    return {
        "prediction": final_label,
        "confidence": round(final_confidence, 2),
    }


def _build_decision_reason(primary_result, verification_result, merged_result):
    final_label = merged_result["prediction"]
    final_conf = merged_result["confidence"]
    fake_conf = primary_result["fake_confidence"]
    real_conf = primary_result["real_confidence"]
    signal_score = primary_result.get("signal_score", 0)

    parts = [
        (
            f"Final decision is {final_label} with {final_conf}% confidence, "
            f"based on the primary result (Fake: {fake_conf}%, Real: {real_conf}%) "
            "and available Gemini/Groq verification."
        )
    ]

    if signal_score >= 5:
        parts.append(
            "The content contains strong fake-news style signals (sensational/viral/exaggerated patterns)."
        )
    elif signal_score >= 2:
        parts.append("The content contains moderate suspicious language patterns.")
    else:
        parts.append("The content does not show strong fake-news style language patterns.")

    verified_results = verification_result.get("results") or []
    if verified_results:
        for result in verified_results:
            parts.append(
                f"{result.get('provider')} returned {result.get('label')} "
                f"with {result.get('confidence')}% confidence."
            )

        for key in ("gemini_result", "groq_result"):
            result = verification_result.get(key) or {}
            if result.get("status") != "verified" and result.get("error"):
                parts.append(f"{result.get('provider')} verification was unavailable.")
    else:
        parts.append("Gemini/Groq verification was unavailable, so the primary result was used.")

    return " ".join(parts)


def _fallback_primary_result(cleaned_text, reason):
    signal_score = _fake_signal_score(cleaned_text)

    # Warm-up mode: keep fake outputs conservative and favor real when
    # suspicious language signals are weak.
    if signal_score >= 6:
        label = "Fake News"
        confidence = 36.0
    elif signal_score >= 3:
        label = "Fake News"
        confidence = 28.0
    else:
        label = "Real News"
        confidence = 72.0

    other_label_confidence = round(100.0 - confidence, 2)
    fake_confidence = confidence if label == "Fake News" else other_label_confidence
    real_confidence = confidence if label == "Real News" else other_label_confidence

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "fake_confidence": round(fake_confidence, 2),
        "real_confidence": round(real_confidence, 2),
        "signal_score": signal_score,
        "model_status": "model_warming",
        "model_note": reason,
    }


def _format_analysis_response(primary_result, verification_result, merged_result, decision_reason):
    return {
        "prediction": merged_result["prediction"],
        "confidence": merged_result["confidence"],
        "decision_reason": decision_reason,
        "primary_prediction": primary_result["label"],
        "primary_confidence": primary_result["confidence"],
        "primary_fake_confidence": primary_result["fake_confidence"],
        "primary_real_confidence": primary_result["real_confidence"],
        "primary_model_status": primary_result.get("model_status", "ready"),
        "primary_model_note": primary_result.get("model_note", ""),
        "signal_score": primary_result.get("signal_score", 0),
        "verification_provider": verification_result.get("provider"),
        "verification_model": verification_result.get("model"),
        "verification_status": verification_result.get("status"),
        "verification_prediction": verification_result.get("label"),
        "verification_confidence": verification_result.get("confidence"),
        "verification_explanation": verification_result.get("explanation"),
        "verification_error": verification_result.get("error", ""),
        "verification_results": verification_result.get("results", []),
        "gemini_result": verification_result.get("gemini_result"),
        "groq_result": verification_result.get("groq_result"),
    }


# def predict_text(text):
#     _load_model()
#     cleaned_text = (text or "").strip()
#     if not cleaned_text:
#         raise ValueError("Empty input text")

#     _model.eval()
#     inputs = _tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=256)
#     with torch.no_grad():
#         outputs = _model(**inputs)

#     probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
#     fake_idx, real_idx = _resolve_indices_from_config(_model)

#     fake_prob = probs[fake_idx].item()
#     real_prob = probs[real_idx].item()

#     # Fallback only for very uncertain outputs.
#     if abs(real_prob - fake_prob) < 0.12 and _fake_signal_score(cleaned_text) >= 2:
#         prediction = fake_idx
#     else:
#         prediction = int(torch.argmax(probs).item())

#     if prediction == real_idx:
#         return "Real News", round(real_prob * 100, 2)
#     return "Fake News", round(fake_prob * 100, 2)

def predict_text(text):
    """
    Optimized prediction with CPU inference.
    ~100-200ms per prediction on low-power devices.
    """
    analysis = analyze_text(text)
    return analysis["prediction"], analysis["confidence"]


def analyze_text(text):
    cleaned_text = (text or "").strip()

    if not cleaned_text:
        raise ValueError("Empty input text")

    if not _is_model_ready():
        load_error = ""
        try:
            _load_model()
        except Exception as exc:
            load_error = str(exc)
            logger.warning("Synchronous model load failed, using warm fallback: %s", exc)
            load_model_background()

    if not _is_model_ready():
        warm_reason = "The full BERT model is still warming up."
        if load_error:
            warm_reason = f"The full BERT model is still warming up ({load_error})."
        primary_result = _fallback_primary_result(
            cleaned_text,
            warm_reason,
        )
        verification_result = _verify_with_llm(
            cleaned_text,
            primary_result["label"],
            primary_result["confidence"],
        )
        merged_result = _merge_predictions(primary_result, verification_result)
        _apply_fake_confidence_caps(primary_result, verification_result, merged_result)
        decision_reason = _build_decision_reason(primary_result, verification_result, merged_result)
        return _format_analysis_response(primary_result, verification_result, merged_result, decision_reason)

    torch = _torch

    inputs = _tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    fake_idx, real_idx = _resolve_indices_from_config(_model)

    fake_prob = probs[fake_idx].item()
    real_prob = probs[real_idx].item()
    signal_score = _fake_signal_score(cleaned_text)

    # Decision policy: keep model-first behavior, but add stronger protection
    # against obvious fake-style language when confidence margin is small.
    confidence_gap = abs(real_prob - fake_prob)

    if signal_score >= 6 and fake_prob >= 0.30:
        prediction = fake_idx
    elif signal_score >= 4 and fake_prob >= 0.35:
        prediction = fake_idx
    elif signal_score >= 2 and fake_prob >= 0.40:
        prediction = fake_idx
    elif signal_score >= 5 and confidence_gap <= 0.22:
        prediction = fake_idx
    elif real_prob > 0.75:
        prediction = real_idx
    elif fake_prob > 0.60:
        prediction = fake_idx
    else:
        prediction = int(torch.argmax(probs).item())

    primary_label = "Real News" if prediction == real_idx else "Fake News"
    primary_confidence = round((real_prob if prediction == real_idx else fake_prob) * 100, 2)

    primary_result = {
        "label": primary_label,
        "confidence": primary_confidence,
        "fake_confidence": round(fake_prob * 100, 2),
        "real_confidence": round(real_prob * 100, 2),
        "signal_score": signal_score,
        "model_status": "ready",
        "model_note": "",
    }
    verification_result = _verify_with_llm(cleaned_text, primary_label, primary_confidence)
    merged_result = _merge_predictions(primary_result, verification_result)
    _apply_fake_confidence_caps(primary_result, verification_result, merged_result)
    decision_reason = _build_decision_reason(primary_result, verification_result, merged_result)

    return _format_analysis_response(primary_result, verification_result, merged_result, decision_reason)


# def extract_text_from_url(url):
#     try:
#         from newspaper import Article
#         from newspaper import Config
#     except ImportError as exc:
#         raise ImportError(
#             "Missing dependency for URL extraction. Install newspaper3k and lxml_html_clean."
#         ) from exc

#     config = Config()
#     config.browser_user_agent = ARTICLE_REQUEST_HEADERS["User-Agent"]
#     config.request_timeout = 15

#     try:
#         article = Article(url, config=config)
#         article.download()
#         article.parse()
#         text = (article.text or "").strip()
#         if text:
#             return text
#     except Exception as exc:
#         logger.warning("newspaper3k extraction failed for %s: %s", url, exc)

#     try:
#         if curl_requests is not None:
#             response = curl_requests.get(
#                 url,
#                 headers=ARTICLE_REQUEST_HEADERS,
#                 timeout=20,
#                 impersonate="chrome123",
#             )
#         else:
#             response = requests.get(url, headers=ARTICLE_REQUEST_HEADERS, timeout=20)
#         response.raise_for_status()
#     except requests.RequestException as exc:
#         raise ValueError(
#             "The news site blocked automated access or the page could not be fetched."
#         ) from exc
#     except Exception as exc:
#         raise ValueError(
#             "The news site blocked automated access or the page could not be fetched."
#         ) from exc

#     soup = BeautifulSoup(response.text, "html.parser")

#     for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside"]):
#         tag.decompose()

#     candidates = []
#     for selector in (
#         "article",
#         "main",
#         "[role='main']",
#         ".article-body",
#         ".story-body",
#         ".entry-content",
#         ".post-content",
#         ".article-content",
#     ):
#         candidates.extend(soup.select(selector))

#     paragraph_texts = []
#     seen_blocks = set()

#     for block in candidates:
#         text = " ".join(
#             p.get_text(" ", strip=True)
#             for p in block.find_all(["p", "h2", "h3", "li"])
#         ).strip()
#         normalized = re.sub(r"\s+", " ", text)
#         if len(normalized) >= 200 and normalized not in seen_blocks:
#             paragraph_texts.append(normalized)
#             seen_blocks.add(normalized)

#     if not paragraph_texts:
#         paragraphs = [
#             re.sub(r"\s+", " ", p.get_text(" ", strip=True))
#             for p in soup.find_all("p")
#         ]
#         paragraphs = [text for text in paragraphs if len(text) >= 40]
#         combined = " ".join(paragraphs).strip()
#         if combined:
#             paragraph_texts.append(combined)

#     extracted_text = "\n\n".join(paragraph_texts).strip()
#     if extracted_text:
#         return extracted_text

#     raise ValueError("The page loaded, but no readable article text was found.")




def extract_text_from_url(url):
    """
    Robust extractor:
    1. newspaper3k parse
    2. curl_cffi browser impersonation on URL variants
    3. requests fallback on URL variants
    4. trafilatura/JSON-LD/DOM parsing fallbacks
    """

    def _url_variants(input_url):
        base = _normalize_article_url(input_url)
        parsed = urlparse(base)
        variants = [base]

        alt_scheme = "http" if parsed.scheme == "https" else "https"
        variants.append(urlunparse(parsed._replace(scheme=alt_scheme)))

        if parsed.netloc.startswith("www."):
            variants.append(urlunparse(parsed._replace(netloc=parsed.netloc[4:])))
        else:
            variants.append(urlunparse(parsed._replace(netloc=f"www.{parsed.netloc}")))

        base = variants[0]
        if "?" in base:
            variants.append(f"{base}&output=amp")
            variants.append(f"{base}&amp=1")
        else:
            variants.append(f"{base}?output=amp")
            variants.append(f"{base}?amp=1")
        if not base.endswith("/amp"):
            variants.append(f"{base.rstrip('/')}/amp")
        max_variants = _get_env_int("URL_MAX_FETCH_VARIANTS", 8)
        return [u for i, u in enumerate(variants) if u and u not in variants[:i]][:max_variants]

    candidate_urls = _url_variants(url)
    fetch_timeout = _get_env_int("URL_FETCH_TIMEOUT", 10)
    curl_requests = _get_curl_requests()
    requests = _get_requests()

    # -------- STEP 1: newspaper3k --------
    try:
        from newspaper import Article, Config
        config = Config()
        config.browser_user_agent = ARTICLE_REQUEST_HEADERS["User-Agent"]
        config.request_timeout = fetch_timeout

        article = Article(candidate_urls[0], config=config)
        article.download()
        article.parse()

        if article.text.strip():
            return article.text.strip()
    except Exception:
        pass

    html = None

    # -------- STEP 2: CURL (BEST BYPASS) --------
    try:
        if curl_requests is not None:
            for candidate_url in candidate_urls:
                for browser_profile in ("chrome124", "chrome123"):
                    response = curl_requests.get(
                        candidate_url,
                        headers=ARTICLE_REQUEST_HEADERS,
                        impersonate=browser_profile,
                        timeout=fetch_timeout,
                        allow_redirects=True,
                    )
                    if 200 <= response.status_code < 400 and response.text:
                        html = response.text
                        break
                if html:
                    break
    except Exception:
        pass

    # -------- STEP 3: requests fallback --------
    if not html:
        for candidate_url in candidate_urls:
            try:
                res = requests.get(
                    candidate_url,
                    headers=ARTICLE_REQUEST_HEADERS,
                    timeout=fetch_timeout,
                    allow_redirects=True,
                )
                res.raise_for_status()
                if res.text:
                    html = res.text
                    break
            except Exception:
                continue

    if not html:
        raise ValueError(
            "The news site blocked automated access or the page could not be fetched."
        )

    # -------- STEP 4: trafilatura fallback --------
    try:
        from importlib import import_module

        trafilatura = import_module("trafilatura")
        extracted = trafilatura.extract(
            html,
            include_tables=False,
            include_comments=False,
            favor_recall=True,
        )
        if extracted and len(extracted.strip()) >= 200:
            return extracted.strip()
    except Exception:
        pass

    # -------- STEP 5: BeautifulSoup parsing --------
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Try JSON-LD article bodies before aggressive cleanup.
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue

        stack = payload if isinstance(payload, list) else [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                article_body = node.get("articleBody")
                if isinstance(article_body, str) and len(article_body.strip()) >= 200:
                    return re.sub(r"\s+", " ", article_body).strip()
                graph = node.get("@graph")
                if isinstance(graph, list):
                    stack.extend(graph)

    for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Universal selectors + site-specific
    selectors = [
        "article",
        "[itemprop='articleBody']",
        "div[itemprop='articleBody']",
        "main",
        "[role='main']",
        ".article-body",
        ".story-body",
        ".entry-content",
        ".post-content",
        ".article-content",
        ".main-story-content",  # WIONews
        ".article-main",
        ".content",
    ]

    candidates = []
    for sel in selectors:
        candidates.extend(soup.select(sel))

    texts = []
    for block in candidates:
        text = " ".join(
            p.get_text(" ", strip=True)
            for p in block.find_all(["p", "h2", "h3", "li"])
        )
        if len(text) > 200:
            texts.append(text)

    # -------- STEP 6: fallback paragraph --------
    if not texts:
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
        ]
        paragraphs = [p for p in paragraphs if len(p) >= 40]
        text = " ".join(paragraphs)
        if len(text) > 200:
            texts.append(text)

    # -------- STEP 7: META fallback --------
    if not texts:
        title = soup.title.string if soup.title else ""
        meta = (
            soup.find("meta", attrs={"name": "description"})
            or soup.find("meta", attrs={"property": "og:description"})
            or soup.find("meta", attrs={"name": "twitter:description"})
        )
        desc = meta.get("content", "") if meta else ""
        fallback = f"{title}. {desc}".strip()

        if fallback:
            return fallback

    final_text = "\n\n".join(texts).strip()

    if final_text:
        return final_text

    raise ValueError("Content not extractable from this site.")


def load_model_background():
    """
    Load model in background thread (call on Django startup).
    This way, the model is ready when first request comes in.
    """
    global _model_load_thread

    with _model_start_lock:
        if _is_model_ready():
            return _model_load_thread
        if _model_load_thread is not None and _model_load_thread.is_alive():
            return _model_load_thread

    def _load():
        try:
            _load_model()
            logger.info("Model pre-loaded in background")
        except Exception as e:
            logger.error(f" Background model loading failed: {e}")
    
    thread = threading.Thread(target=_load, daemon=True)
    _model_load_thread = thread
    thread.start()
    return thread
