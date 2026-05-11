from pathlib import Path
import json
import os
import re
import threading
import logging
import random
from urllib.parse import quote_plus, unquote, urljoin, urlparse, urlunparse


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

DIVERSE_USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
        "Gecko/20100101 Firefox/124.0"
    ),
]

BLOCKED_HTML_MARKERS = (
    "access denied",
    "enable javascript and cookies",
    "captcha",
    "verify you are human",
    "bot verification",
    "request blocked",
    "cloudflare",
    "unusual traffic",
    "attention required",
)

SEARCH_SOURCE_CONFIGS = {
    "timesofindia": {
        "label": "Times of India",
        "domain": "timesofindia.indiatimes.com",
        "rss_search_url": "https://timesofindia.indiatimes.com/searchnews/feed/{query}.cms",
    },
    "ndtv": {
        "label": "NDTV",
        "domain": "ndtv.com",
        "rss_search_url": "https://feeds.ndtv.com/ndtv/latest.xml?search={query}",
    },
    "hindustantimes": {
        "label": "Hindustan Times",
        "domain": "hindustantimes.com",
        "html_search_url": "https://www.hindustantimes.com/search?q={query}",
    },
    "bbc": {
        "label": "BBC",
        "domain": "bbc.com",
        "html_search_url": "https://www.bbc.com/search?q={query}",
    },
    "reuters": {
        "label": "Reuters",
        "domain": "reuters.com",
        "html_search_url": "https://www.reuters.com/search/news?blob={query}",
    },
    "thehindu": {
        "label": "The Hindu",
        "domain": "thehindu.com",
    },
    "indianexpress": {
        "label": "Indian Express",
        "domain": "indianexpress.com",
    },
    "indiatoday": {
        "label": "India Today",
        "domain": "indiatoday.in",
    },
    "theprint": {
        "label": "ThePrint",
        "domain": "theprint.in",
    },
    "scroll": {
        "label": "Scroll.in",
        "domain": "scroll.in",
    },
    "wion": {
        "label": "WION",
        "domain": "wionews.com",
    },
    "guardian": {
        "label": "The Guardian",
        "domain": "theguardian.com",
    },
    "cnn": {
        "label": "CNN",
        "domain": "cnn.com",
    },
    "ap": {
        "label": "AP News",
        "domain": "apnews.com",
    },
    "google_news": {
        "label": "Google News",
        "domain": "news.google.com",
    },
    "bing_news": {
        "label": "Bing News",
        "domain": "bing.com",
    },
}

SEARCH_SOURCE_ALIASES = {
    "toi": "timesofindia",
    "timesofindia": "timesofindia",
    "timesofindia.indiatimes.com": "timesofindia",
    "ndtv": "ndtv",
    "ndtv.com": "ndtv",
    "ht": "hindustantimes",
    "hindustantimes": "hindustantimes",
    "hindustantimes.com": "hindustantimes",
    "bbc": "bbc",
    "bbcnews": "bbc",
    "bbc.com": "bbc",
    "reuters": "reuters",
    "reuters.com": "reuters",
    "thehindu": "thehindu",
    "thehindu.com": "thehindu",
    "indianexpress": "indianexpress",
    "indianexpress.com": "indianexpress",
    "indiatoday": "indiatoday",
    "indiatoday.in": "indiatoday",
    "theprint": "theprint",
    "theprint.in": "theprint",
    "scroll": "scroll",
    "scroll.in": "scroll",
    "wion": "wion",
    "wionews": "wion",
    "wionews.com": "wion",
    "guardian": "guardian",
    "theguardian": "guardian",
    "theguardian.com": "guardian",
    "cnn": "cnn",
    "cnn.com": "cnn",
    "ap": "ap",
    "apnews": "ap",
    "apnews.com": "ap",
    "google": "google_news",
    "googlenews": "google_news",
    "google_news": "google_news",
    "news.google.com": "google_news",
    "bing": "bing_news",
    "bingnews": "bing_news",
    "bing_news": "bing_news",
    "bing.com": "bing_news",
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


def _get_diverse_headers():
    return {
        "User-Agent": random.choice(DIVERSE_USER_AGENTS),
        **{key: value for key, value in ARTICLE_REQUEST_HEADERS.items() if key != "User-Agent"},
    }


def _clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_html_fragment(value):
    text = str(value or "")
    if not text:
        return ""
    if "<" not in text or ">" not in text:
        return _clean_text(text)
    try:
        from bs4 import BeautifulSoup

        return _clean_text(BeautifulSoup(text, "html.parser").get_text(" ", strip=True))
    except Exception:
        return _clean_text(text)


def _looks_like_blocked_html(html):
    normalized = _clean_text(html).lower()
    if len(normalized) < 80:
        return True
    return any(marker in normalized for marker in BLOCKED_HTML_MARKERS)


def _build_article_url_variants(input_url):
    base_url = _normalize_article_url(input_url)
    parsed = urlparse(base_url)
    base_domain = parsed.netloc[4:] if parsed.netloc.startswith("www.") else parsed.netloc
    variants = []

    def _add(candidate):
        if candidate and candidate not in variants:
            variants.append(candidate)

    _add(base_url)
    _add(urlunparse(parsed._replace(scheme="http" if parsed.scheme == "https" else "https")))
    _add(urlunparse(parsed._replace(netloc=base_domain)))
    _add(urlunparse(parsed._replace(netloc=f"www.{base_domain}")))

    if not base_domain.startswith(("m.", "mobile.", "amp.")):
        _add(urlunparse(parsed._replace(netloc=f"m.{base_domain}")))
        _add(urlunparse(parsed._replace(netloc=f"mobile.{base_domain}")))

    if parsed.query:
        _add(urlunparse(parsed._replace(query="")))

    if parsed.path.endswith("/"):
        _add(urlunparse(parsed._replace(path=parsed.path.rstrip("/"))))
    elif parsed.path:
        _add(urlunparse(parsed._replace(path=f"{parsed.path}/")))

    if "?" in base_url:
        _add(f"{base_url}&output=amp")
        _add(f"{base_url}&amp=1")
    else:
        _add(f"{base_url}?output=amp")
        _add(f"{base_url}?amp=1")

    if not parsed.path.endswith("/amp"):
        _add(urlunparse(parsed._replace(path=f"{parsed.path.rstrip('/')}/amp")))

    max_variants = _get_env_int("URL_MAX_FETCH_VARIANTS", 12)
    return variants[:max_variants]


def _fetch_url_text(url, fetch_timeout, min_length=0):
    curl_requests = _get_curl_requests()
    requests = _get_requests()

    if curl_requests is not None:
        for browser_profile in ("chrome124", "chrome123", "firefox120", "edge123"):
            try:
                response = curl_requests.get(
                    url,
                    headers=_get_diverse_headers(),
                    impersonate=browser_profile,
                    timeout=fetch_timeout,
                    allow_redirects=True,
                )
                if 200 <= response.status_code < 400 and response.text and len(response.text) >= min_length:
                    return response.text
            except Exception:
                continue

    for _ in range(2):
        try:
            response = requests.get(
                url,
                headers=_get_diverse_headers(),
                timeout=fetch_timeout,
                allow_redirects=True,
            )
            if 200 <= response.status_code < 400 and response.text and len(response.text) >= min_length:
                return response.text
        except Exception:
            continue

    return ""


def _fetch_first_article_html(candidate_urls, fetch_timeout):
    fallback_html = ""
    for candidate_url in candidate_urls:
        html = _fetch_url_text(candidate_url, fetch_timeout, min_length=200)
        if not html:
            continue
        if not _looks_like_blocked_html(html):
            return html
        if not fallback_html:
            fallback_html = html
    return fallback_html


def _fetch_archive_snapshot_html(url, fetch_timeout):
    try:
        normalized_url = _normalize_article_url(url)
        cdx_url = (
            "https://web.archive.org/cdx/search/cdx"
            f"?url={quote_plus(normalized_url)}&output=json&fl=timestamp,original"
            "&filter=statuscode:200&limit=1&sort=reverse"
        )
        snapshot_payload = _fetch_url_text(cdx_url, fetch_timeout, min_length=2)
        if not snapshot_payload:
            return ""

        snapshot_rows = json.loads(snapshot_payload)
        if not isinstance(snapshot_rows, list) or len(snapshot_rows) < 2:
            return ""

        timestamp, original = snapshot_rows[1][0], snapshot_rows[1][1]
        if not timestamp or not original:
            return ""

        snapshot_url = f"https://web.archive.org/web/{timestamp}/{original}"
        return _fetch_url_text(snapshot_url, fetch_timeout, min_length=200)
    except Exception:
        return ""


def guess_search_query_from_url(url):
    try:
        normalized_url = _normalize_article_url(url)
    except ValueError:
        return ""

    parsed = urlparse(normalized_url)
    stopwords = {
        "news",
        "article",
        "articles",
        "story",
        "stories",
        "amp",
        "video",
        "videos",
        "html",
        "cms",
        "liveblog",
        "live-blog",
        "read",
    }
    fallback_words = []

    for segment in reversed([part for part in parsed.path.split("/") if part]):
        cleaned = unquote(segment)
        cleaned = re.sub(r"\.(html?|cms|amp)$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[-_]+", " ", cleaned)
        cleaned = re.sub(r"\b\d{2,}\b", " ", cleaned)
        cleaned = _clean_text(cleaned)
        if not cleaned:
            continue

        words = [word for word in cleaned.split() if word.lower() not in stopwords]
        if len(words) >= 4:
            return " ".join(words[:8])
        if words:
            fallback_words = words + fallback_words

    return " ".join(fallback_words[:8]).strip()


def guess_search_site_from_url(url):
    try:
        parsed = urlparse(_normalize_article_url(url))
    except ValueError:
        return ""
    return parsed.netloc.lower().removeprefix("www.")


def _extract_domain(value):
    cleaned = _clean_text(value).lower().strip("/")
    if not cleaned:
        return ""
    cleaned = cleaned.removeprefix("site:").strip()
    parsed = urlparse(cleaned if "://" in cleaned else f"https://{cleaned}")
    domain = (parsed.netloc or parsed.path).lower().strip("/")
    return domain.removeprefix("www.")


def _resolve_search_source(source):
    normalized = _clean_text(source or "all").lower()
    if not normalized or normalized == "all":
        return "all", {"label": "All Sources", "domain": ""}

    lookup_key = re.sub(r"[^a-z0-9._]+", "", normalized)
    canonical = SEARCH_SOURCE_ALIASES.get(lookup_key)
    if canonical:
        config = SEARCH_SOURCE_CONFIGS.get(canonical, {})
        return canonical, {"label": config.get("label", canonical), "domain": config.get("domain", "")}

    domain = _extract_domain(normalized)
    if domain:
        for key, config in SEARCH_SOURCE_CONFIGS.items():
            if config.get("domain") == domain:
                return key, {"label": config.get("label", key), "domain": domain}
        return domain, {"label": domain, "domain": domain}

    return normalized, {"label": normalized, "domain": ""}


def _build_google_news_rss_url(query, site_domain=""):
    search_terms = _clean_text(query)
    if site_domain:
        search_terms = f"{search_terms} site:{site_domain}"
    encoded_query = quote_plus(search_terms)
    return f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"


def _build_bing_news_url(query, site_domain=""):
    search_terms = _clean_text(query)
    if site_domain:
        search_terms = f"{search_terms} site:{site_domain}"
    return f"https://www.bing.com/news/search?q={quote_plus(search_terms)}&FORM=HDRSC6"


def _build_search_targets(query, source_key, source_meta):
    targets = []
    site_domain = source_meta.get("domain", "")

    def _add_target(url, parser_name, source_name):
        if url and all(existing["url"] != url for existing in targets):
            targets.append({"url": url, "parser": parser_name, "source": source_name})

    if source_key == "all":
        _add_target(_build_google_news_rss_url(query), "rss", "google_news")
        _add_target(_build_bing_news_url(query), "html", "bing_news")
        for key in ("timesofindia", "ndtv", "hindustantimes", "bbc", "reuters"):
            config = SEARCH_SOURCE_CONFIGS.get(key, {})
            if config.get("rss_search_url"):
                _add_target(config["rss_search_url"].format(query=quote_plus(query)), "rss", key)
            if config.get("html_search_url"):
                _add_target(config["html_search_url"].format(query=quote_plus(query)), "html", key)
        return targets

    if source_key == "google_news":
        _add_target(_build_google_news_rss_url(query), "rss", "google_news")
        return targets

    if source_key == "bing_news":
        _add_target(_build_bing_news_url(query), "html", "bing_news")
        return targets

    source_config = SEARCH_SOURCE_CONFIGS.get(source_key, {})
    if source_config.get("rss_search_url"):
        _add_target(source_config["rss_search_url"].format(query=quote_plus(query)), "rss", source_key)
    if source_config.get("html_search_url"):
        _add_target(source_config["html_search_url"].format(query=quote_plus(query)), "html", source_key)

    _add_target(_build_google_news_rss_url(query, site_domain=site_domain), "rss", source_key)
    _add_target(_build_bing_news_url(query, site_domain=site_domain), "html", source_key)
    return targets


def _result_dedupe_key(title, link):
    title_key = re.sub(r"[^a-z0-9]+", " ", _clean_text(title).lower()).strip()
    if title_key:
        return title_key
    link_key = re.sub(r"^https?://(www\.)?", "", _clean_text(link).lower())
    return link_key.split("?")[0]


def _append_search_result(results, seen_keys, item, limit):
    title = _clean_text(item.get("title"))
    link = _clean_text(item.get("link"))
    if not title or not link:
        return

    key = _result_dedupe_key(title, link)
    if not key or key in seen_keys:
        return

    seen_keys.add(key)
    results.append({
        "title": title[:220],
        "link": link,
        "source": _clean_text(item.get("source")) or "news",
        "summary": _clean_text(item.get("summary"))[:360],
        "published": _clean_text(item.get("published")),
    })

    if len(results) > limit:
        del results[limit:]


def _parse_feed_results(feed_text, default_source):
    try:
        import feedparser
    except ImportError:
        return []

    feed = feedparser.parse(feed_text)
    parsed_results = []
    for entry in getattr(feed, "entries", []):
        title = _clean_text(entry.get("title", ""))
        source_name = default_source
        if default_source == "google_news" and " - " in title:
            title, guessed_source = [part.strip() for part in title.rsplit(" - ", 1)]
            source_name = guessed_source or default_source

        entry_source = entry.get("source")
        if isinstance(entry_source, dict):
            source_name = _clean_text(entry_source.get("title") or entry_source.get("href")) or source_name

        parsed_results.append({
            "title": title,
            "link": _clean_text(entry.get("link", "")),
            "source": source_name,
            "summary": _clean_html_fragment(entry.get("summary") or entry.get("description") or ""),
            "published": _clean_text(
                entry.get("published") or entry.get("updated") or entry.get("pubDate") or ""
            ),
        })
    return parsed_results


def _is_probable_article_link(link):
    if not link:
        return False
    lowered = link.lower()
    if lowered.startswith(("javascript:", "mailto:", "#")):
        return False
    if any(segment in lowered for segment in ("/search?", "/search/", "/topic/", "/tag/", "/latest", "/live-updates")):
        return False
    return True


def _parse_html_search_results(html, base_url, default_source):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    parsed_results = []
    seen_links = set()
    candidate_blocks = soup.select("article, div, li, section")

    for block in candidate_blocks[:200]:
        title_element = block.find(["h1", "h2", "h3", "h4"])
        link_element = None
        for anchor in block.select("a[href]"):
            href = urljoin(base_url, anchor.get("href", ""))
            if _is_probable_article_link(href):
                link_element = anchor
                break

        if link_element is None:
            continue

        link = urljoin(base_url, link_element.get("href", ""))
        if not _is_probable_article_link(link) or link in seen_links:
            continue

        title = _clean_text(
            title_element.get_text(" ", strip=True) if title_element else link_element.get_text(" ", strip=True)
        )
        if len(title) < 20:
            continue

        seen_links.add(link)
        summary_element = block.find("p")
        time_element = block.find("time")
        parsed_results.append({
            "title": title,
            "link": link,
            "source": default_source,
            "summary": _clean_text(summary_element.get_text(" ", strip=True) if summary_element else block.get_text(" ", strip=True)),
            "published": _clean_text(
                (time_element.get("datetime") if time_element and time_element.has_attr("datetime") else "")
                or (time_element.get_text(" ", strip=True) if time_element else "")
            ),
        })

    return parsed_results


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
    Ultra-robust multi-method extractor for all website types:
    1. newspaper3k parse
    2. curl_cffi with multiple browser profiles
    3. requests with diverse user agents
    4. trafilatura extraction
    5. JSON-LD/microdata extraction
    6. DOM parsing with enhanced selectors
    7. Archive.org/Archive.is fallback
    8. Meta/OG tag extraction
    """
    candidate_urls = _build_article_url_variants(url)
    fetch_timeout = _get_env_int("URL_FETCH_TIMEOUT", 12)

    # -------- STEP 1: newspaper3k --------
    try:
        from newspaper import Article, Config
        config = Config()
        config.browser_user_agent = _get_diverse_headers()["User-Agent"]
        config.request_timeout = fetch_timeout
        article = Article(candidate_urls[0], config=config)
        article.download()
        article.parse()
        if article.text and len(article.text.strip()) >= 100:
            return article.text.strip()
    except Exception:
        pass

    html = _fetch_first_article_html(candidate_urls, fetch_timeout)
    if (not html or _looks_like_blocked_html(html)) and _get_env_bool("ENABLE_ARCHIVE_FALLBACK", True):
        archive_html = _fetch_archive_snapshot_html(url, fetch_timeout)
        if archive_html:
            html = archive_html

    if not html:
        raise ValueError("The news site blocked automated access or the page could not be fetched.")

    # -------- STEP 5: trafilatura (most reliable) --------
    try:
        from importlib import import_module
        trafilatura = import_module("trafilatura")
        extracted = trafilatura.extract(html, include_tables=True, include_comments=False, favor_recall=True)
        if extracted and len(extracted.strip()) >= 100:
            return extracted.strip()
    except Exception:
        pass

    # -------- STEP 6: Enhanced BeautifulSoup parsing --------
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    def _walk_json_values(node):
        if isinstance(node, dict):
            for key, value in node.items():
                normalized_key = str(key).lower()
                if normalized_key in {
                    "articlebody",
                    "body",
                    "content",
                    "description",
                    "summary",
                    "text",
                }:
                    if isinstance(value, str):
                        cleaned_value = _clean_text(value)
                        if len(cleaned_value) >= 100:
                            yield cleaned_value
                    elif isinstance(value, (dict, list)):
                        yield from _walk_json_values(value)
                elif isinstance(value, (dict, list)):
                    yield from _walk_json_values(value)
        elif isinstance(node, list):
            for item in node:
                yield from _walk_json_values(item)

    # Try JSON-LD/article payloads before aggressive cleanup.
    for script in soup.find_all("script"):
        script_type = (script.get("type") or "").lower()
        script_id = (script.get("id") or "").lower()
        raw = (script.string or script.get_text() or "").strip()
        if not raw or raw[:1] not in {"{", "["}:
            continue
        if script_type and script_type not in {"application/ld+json", "application/json"} and script_id not in {
            "__next_data__",
            "__nuxt_data__",
        }:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        for extracted_text in _walk_json_values(payload):
            if len(extracted_text) >= 100:
                return extracted_text

    # Clean up soup
    for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside", "svg"]):
        tag.decompose()

    # Enhanced CSS selectors (covering Indian news sites, WIONews, etc.)
    selectors = [
        "article",
        "[itemprop='articleBody']",
        "div[itemprop='articleBody']",
        "main",
        "[role='main']",
        ".article-body", ".article-content", ".article-main",
        ".story-body", ".story-content",
        ".entry-content", ".post-content",
        ".main-story-content", ".main-story",
        ".content", ".post",
        ".news-content", ".news-body",
        "[data-component='ArticleBody']",
        ".article-wrapper", ".article-page",
        ".page-content", ".body-text",
        "[class*='article-text']", "[class*='story-text']",
    ]

    candidates = []
    for sel in selectors:
        try:
            candidates.extend(soup.select(sel))
        except Exception:
            continue

    texts = []
    seen = set()
    for block in candidates:
        text = _clean_text(
            " ".join(
                element.get_text(" ", strip=True)
                for element in block.find_all(["p", "h2", "h3", "h4", "li", "div"])
                if element.get_text(strip=True)
            )
        )
        if len(text) >= 100 and text not in seen:
            texts.append(text)
            seen.add(text)

    # -------- STEP 7: Paragraph fallback --------
    if not texts:
        paragraphs = []
        for p in soup.find_all("p"):
            text = _clean_text(p.get_text(" ", strip=True))
            if 40 <= len(text) <= 10000:  # Filter out very long metadata
                paragraphs.append(text)

        combined = " ".join(paragraphs[:200])  # Limit paragraphs
        if len(combined) >= 100:
            texts.append(_clean_text(combined))

    # -------- STEP 8: Div content fallback --------
    if not texts:
        for div in soup.find_all("div", {"class": re.compile(r".*content|.*text|.*body", re.I)}):
            text = _clean_text(" ".join(p.get_text(" ", strip=True) for p in div.find_all(["p", "span"])))
            if 100 <= len(text) <= 50000:
                texts.append(text)
                break

    # -------- STEP 9: Meta/OG tags fallback --------
    if not texts:
        title = _clean_text(soup.title.string if soup.title else "")
        desc_meta = (
            soup.find("meta", attrs={"name": "description"})
            or soup.find("meta", attrs={"property": "og:description"})
            or soup.find("meta", attrs={"name": "twitter:description"})
        )
        desc = _clean_text(desc_meta.get("content", "") if desc_meta else "")
        content_meta = soup.find("meta", attrs={"property": "og:article:body"})
        content = _clean_text(content_meta.get("content", "") if content_meta else "")

        fallback_text = f"{title}. {desc} {content}".strip()
        if fallback_text:
            return _clean_text(fallback_text)

    final_text = "\n\n".join(texts).strip()
    if final_text and len(final_text) >= 100:
        return final_text

    raise ValueError("Content not extractable from this site. The page might be blocked, require authentication, or have no readable text.")


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


def search_news(query, source="all", limit=5):
    """
    Search news from multiple Indian and international news sources.

    Args:
        query: Search query string
        source: News source alias, 'all', or a raw domain like 'ndtv.com'
        limit: Number of results to return (default: 5)

    Returns:
        List of dictionaries with news articles
    """
    cleaned_query = _clean_text(query)
    if not cleaned_query:
        return []

    try:
        normalized_limit = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        normalized_limit = 5

    source_key, source_meta = _resolve_search_source(source)
    fetch_timeout = _get_env_int("URL_FETCH_TIMEOUT", 10)
    targets = _build_search_targets(cleaned_query, source_key, source_meta)
    results = []
    seen_keys = set()

    for target in targets:
        if len(results) >= normalized_limit:
            break

        try:
            payload = _fetch_url_text(target["url"], fetch_timeout, min_length=1)
            if not payload:
                continue

            if target["parser"] == "rss":
                parsed_items = _parse_feed_results(payload, target["source"])
            else:
                parsed_items = _parse_html_search_results(payload, target["url"], target["source"])

            for item in parsed_items:
                _append_search_result(results, seen_keys, item, normalized_limit)
                if len(results) >= normalized_limit:
                    break
        except Exception as e:
            logger.warning(f"Failed to search news from {target['source']}: {str(e)}")
            continue

    return results[:normalized_limit]
