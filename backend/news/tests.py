import json
from io import BytesIO
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase

from news.utils import _apply_groq_display_boost, _build_decision_reason


def _sample_analysis():
    return {
        "prediction": "Fake News",
        "confidence": 91.5,
        "decision_reason": "Suspicious wording and high fake-news signal score.",
        "signal_score": 6,
        "primary_prediction": "Fake News",
        "primary_confidence": 89.2,
        "primary_fake_confidence": 89.2,
        "primary_real_confidence": 10.8,
        "primary_model_status": "ready",
        "primary_model_note": "",
        "verification_status": "skipped",
        "verification_provider": None,
        "verification_model": None,
        "verification_prediction": None,
        "verification_confidence": None,
        "verification_explanation": "",
        "verification_error": "",
        "verification_results": [],
        "gemini_result": None,
        "groq_result": None,
    }


def _build_test_image():
    from PIL import Image

    image = Image.new("RGB", (240, 120), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class PredictNewsApiTests(TestCase):
    @patch("news.utils.analyze_text")
    def test_predict_news_accepts_manual_text(self, mock_analyze_text):
        mock_analyze_text.return_value = _sample_analysis()

        response = self.client.post(
            "/api/predict/",
            data=json.dumps({"text": "This is a sample article."}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["prediction"], "Fake News")
        self.assertEqual(payload["input_source"], "text")
        self.assertEqual(payload["text"], "This is a sample article.")
        mock_analyze_text.assert_called_once_with("This is a sample article.")

    @patch("news.utils.analyze_text")
    @patch("news.utils.extract_text_from_image")
    def test_predict_news_accepts_image_upload(self, mock_extract_text_from_image, mock_analyze_text):
        mock_extract_text_from_image.return_value = {
            "text": "Breaking headline captured from image.",
            "engine": "Windows.Media.Ocr",
            "language": "en",
            "image_name": "headline.png",
        }
        mock_analyze_text.return_value = _sample_analysis()

        upload = SimpleUploadedFile(
            "headline.png",
            _build_test_image(),
            content_type="image/png",
        )

        response = self.client.post("/api/predict/", data={"image": upload})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["input_source"], "image")
        self.assertEqual(payload["image_name"], "headline.png")
        self.assertEqual(payload["ocr_engine"], "Windows.Media.Ocr")
        self.assertEqual(payload["text"], "Breaking headline captured from image.")
        mock_extract_text_from_image.assert_called_once()
        mock_analyze_text.assert_called_once_with("Breaking headline captured from image.")

    def test_predict_news_requires_text_url_or_image(self):
        response = self.client.post(
            "/api/predict/",
            data=json.dumps({}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "No text, URL, or image provided")


class AnalysisFormattingTests(SimpleTestCase):
    def test_groq_display_boost_adds_two_percent_for_matching_label(self):
        merged_result = {"prediction": "Fake News", "confidence": 86.89}
        verification_result = {
            "groq_result": {
                "status": "verified",
                "label": "Fake News",
                "confidence": 92.16,
            }
        }

        _apply_groq_display_boost(verification_result, merged_result)

        self.assertEqual(merged_result["confidence"], 94.16)

    def test_decision_reason_hides_provider_details(self):
        reason = _build_decision_reason(
            {"signal_score": 1},
            {"groq_result": {"provider": "Groq"}, "gemini_result": {"provider": "Gemini"}},
            {"prediction": "Fake News", "confidence": 94.16},
        )

        self.assertIn("Final decision is Fake News with 94.16% confidence.", reason)
        self.assertNotIn("Groq", reason)
        self.assertNotIn("Gemini", reason)
