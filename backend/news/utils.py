from pathlib import Path
import json
import os
import re
import threading
import logging

import requests
import torch
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from curl_cffi import requests as curl_requests
except ImportError:  # pragma: no cover - optional fallback client
    curl_requests = None


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
_model_lock = threading.Lock()
_model_loaded = False


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
    global _tokenizer, _model, _model_loaded
    
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
            
            logger.info("Model loaded successfully on CPU")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        finally:
            _model_loaded = True


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


def _get_env_value(*keys):
    for key in keys:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return ""


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

    response = requests.post(
        endpoint,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        json=payload,
        timeout=20,
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

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=20,
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


def _verify_with_llm(text, primary_label, primary_confidence):
    verifiers = (
        _call_gemini_verifier,
        _call_groq_verifier,
    )

    last_error = ""
    for verifier in verifiers:
        try:
            result = verifier(text, primary_label, primary_confidence)
            result["status"] = "verified"
            return result
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Verifier %s failed: %s", verifier.__name__, exc)

    return {
        "provider": None,
        "model": None,
        "label": primary_label,
        "confidence": primary_confidence,
        "explanation": "",
        "status": "skipped",
        "error": last_error or (
            f"No verifier API key configured. Add keys in {ENV_PATH.name} to enable Gemini/Groq verification."
        ),
    }


def _merge_predictions(primary_result, verification_result):
    primary_fake = primary_result["fake_confidence"]
    primary_real = primary_result["real_confidence"]

    if verification_result.get("status") != "verified":
        final_label = primary_result["label"]
        final_confidence = primary_result["confidence"]
    else:
        verifier_confidence = verification_result["confidence"]
        verifier_fake = verifier_confidence if verification_result["label"] == "Fake News" else 100 - verifier_confidence
        verifier_real = verifier_confidence if verification_result["label"] == "Real News" else 100 - verifier_confidence

        fake_score = (primary_fake * 0.65) + (verifier_fake * 0.35)
        real_score = (primary_real * 0.65) + (verifier_real * 0.35)

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
            f"based on primary model scores (Fake: {fake_conf}%, Real: {real_conf}%)."
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

    if verification_result.get("status") == "verified":
        parts.append(
            f"External verification by {verification_result.get('provider')} also returned "
            f"{verification_result.get('label')} with {verification_result.get('confidence')}% confidence."
        )
    else:
        parts.append("External verification was unavailable, so the primary model decision was used.")

    return " ".join(parts)


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
    _load_model()
    cleaned_text = (text or "").strip()

    if not cleaned_text:
        raise ValueError("Empty input text")

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
    }
    verification_result = _verify_with_llm(cleaned_text, primary_label, primary_confidence)
    merged_result = _merge_predictions(primary_result, verification_result)
    decision_reason = _build_decision_reason(primary_result, verification_result, merged_result)

    return {
        "prediction": merged_result["prediction"],
        "confidence": merged_result["confidence"],
        "decision_reason": decision_reason,
        "primary_prediction": primary_label,
        "primary_confidence": primary_confidence,
        "primary_fake_confidence": primary_result["fake_confidence"],
        "primary_real_confidence": primary_result["real_confidence"],
        "signal_score": signal_score,
        "verification_provider": verification_result.get("provider"),
        "verification_model": verification_result.get("model"),
        "verification_status": verification_result.get("status"),
        "verification_prediction": verification_result.get("label"),
        "verification_confidence": verification_result.get("confidence"),
        "verification_explanation": verification_result.get("explanation"),
        "verification_error": verification_result.get("error", ""),
    }


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
        base = (input_url or "").strip()
        variants = [base]
        if "?" in base:
            variants.append(f"{base}&output=amp")
            variants.append(f"{base}&amp=1")
        else:
            variants.append(f"{base}?output=amp")
            variants.append(f"{base}?amp=1")
        if not base.endswith("/amp"):
            variants.append(f"{base.rstrip('/')}/amp")
        return [u for i, u in enumerate(variants) if u and u not in variants[:i]]

    # -------- STEP 1: newspaper3k --------
    try:
        from newspaper import Article, Config
        config = Config()
        config.browser_user_agent = ARTICLE_REQUEST_HEADERS["User-Agent"]
        config.request_timeout = 10

        article = Article(url, config=config)
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
            for candidate_url in _url_variants(url):
                for browser_profile in ("chrome124", "chrome123", "safari15_5"):
                    response = curl_requests.get(
                        candidate_url,
                        headers=ARTICLE_REQUEST_HEADERS,
                        impersonate=browser_profile,
                        timeout=20,
                        allow_redirects=True,
                    )
                    if response.status_code == 200 and response.text:
                        html = response.text
                        break
                if html:
                    break
    except Exception:
        pass

    # -------- STEP 3: requests fallback --------
    if not html:
        for candidate_url in _url_variants(url):
            try:
                res = requests.get(
                    candidate_url,
                    headers=ARTICLE_REQUEST_HEADERS,
                    timeout=20,
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
    def _load():
        try:
            _load_model()
            logger.info("Model pre-loaded in background")
        except Exception as e:
            logger.error(f" Background model loading failed: {e}")
    
    thread = threading.Thread(target=_load, daemon=True)
    thread.start()
    return thread
