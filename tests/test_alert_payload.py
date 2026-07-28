import unittest
from unittest.mock import patch

from model.blacklist import check_transcript
from model.monitored_terms import classify_transcript_language
from sender.http_client import build_alert_payload, send_alert


class AlertPayloadTests(unittest.TestCase):
    def _payload(self, **overrides):
        values = {
            "severity": "medium",
            "confidence": 0.87,
            "duration": 2.5,
            "transcribed_text": "example speech",
            "detected_words": ["example monitored phrase"],
            "categories": ["emotional_taunting"],
            "language": "ceb",
            "language_confidence": 0.82,
            "matched_terms": [{
                "term_id": 12,
                "term": "example monitored phrase",
                "language": "ceb",
                "match_type": "phrase",
            }],
            "hard_hits": [],
            "soft_hits": ["example monitored phrase"],
            "required_duration": 3.0,
            "duration_gate": "medium",
            "yamnet_class": "Speech",
            "yamnet_score": 0.6,
            "emotion": "upset",
            "tone_data": {
                "rms": 250,
                "energy_variance": 2000,
                "zero_crossing_rate": 0.12,
                "peak_to_average": 3.2,
            },
            "waveform_snapshot": [1, 2, 3],
        }
        values.update(overrides)
        return build_alert_payload(**values)

    def test_alert_payload_contract(self):
        payload = self._payload()
        self.assertEqual(payload["transcript"], "example speech")
        self.assertEqual(payload["transcribed_text"], "example speech")
        self.assertEqual(payload["language"], "ceb")
        self.assertEqual(payload["language_confidence"], 0.82)
        self.assertEqual(
            payload["matched_terms"],
            [{
                "term_id": 12,
                "term": "example monitored phrase",
                "language": "ceb",
                "match_type": "phrase",
            }],
        )

        # Existing production fields remain present.
        for field in (
            "severity", "confidence", "duration", "required_duration",
            "duration_gate", "location", "detected_words", "categories",
            "hard_hits", "soft_hits", "yamnet_class", "yamnet_score",
            "emotion", "rms", "energy_variance", "zero_crossing_rate",
            "peak_to_average", "waveform_snapshot",
        ):
            self.assertIn(field, payload)

    def test_null_confidence_and_local_fallback_without_invented_id(self):
        payload = self._payload(
            language="mixed",
            language_confidence=None,
            matched_terms=[{
                "term": "bobo kaayo ka",
                "language": "mixed",
                "match_type": "phrase",
            }],
        )
        self.assertIsNone(payload["language_confidence"])
        self.assertNotIn("term_id", payload["matched_terms"][0])
        self.assertIsNone(payload["matched_terms"][0]["language"])

    def test_top_level_mixed_with_matched_term_language_null(self):
        payload = self._payload(
            language="mixed",
            matched_terms=[{
                "term": "bobo kaayo ka",
                "language": "mixed",
                "match_type": "phrase",
            }],
        )
        self.assertEqual(payload["language"], "mixed")
        self.assertIsNone(payload["matched_terms"][0]["language"])

    def test_top_level_unknown_with_matched_term_language_null(self):
        payload = self._payload(
            language="unknown",
            matched_terms=[{
                "term": "unclassified fallback",
                "language": "unknown",
                "match_type": "phrase",
            }],
        )
        self.assertEqual(payload["language"], "unknown")
        self.assertIsNone(payload["matched_terms"][0]["language"])

    def test_top_level_ceb_with_matched_term_language_ceb(self):
        payload = self._payload(
            language="ceb",
            matched_terms=[{
                "term_id": 12,
                "term": "example monitored phrase",
                "language": "ceb",
                "match_type": "phrase",
            }],
        )
        self.assertEqual(payload["language"], "ceb")
        self.assertEqual(payload["matched_terms"][0]["language"], "ceb")

    def test_mixed_transcript_keeps_separate_specific_term_languages(self):
        result = check_transcript("Ikaw ay bobo, buang kaayo ka.")
        language, confidence = classify_transcript_language(
            "tl",
            0.91,
            result["matched_terms"],
        )
        payload = self._payload(
            language=language,
            language_confidence=confidence,
            matched_terms=result["matched_terms"],
        )
        self.assertEqual(payload["language"], "mixed")
        self.assertEqual(
            {item["language"] for item in payload["matched_terms"]},
            {"fil", "ceb"},
        )

    def test_backend_dictionary_term_language_is_preserved(self):
        payload = self._payload(
            matched_terms=[{
                "term_id": 12,
                "term": "example monitored phrase",
                "language": "ceb",
                "match_type": "phrase",
            }],
        )
        self.assertEqual(payload["matched_terms"][0]["term_id"], 12)
        self.assertEqual(payload["matched_terms"][0]["language"], "ceb")

    def test_no_matched_term_emits_mixed_or_unknown(self):
        payload = self._payload(matched_terms=[
            {
                "term": "mixed fallback",
                "language": "mixed",
                "match_type": "phrase",
            },
            {
                "term": "unknown fallback",
                "language": "unknown",
                "match_type": "phrase",
            },
            {
                "term": "specific phrase",
                "language": "fil",
                "match_type": "phrase",
            },
        ])
        languages = [item["language"] for item in payload["matched_terms"]]
        self.assertNotIn("mixed", languages)
        self.assertNotIn("unknown", languages)
        self.assertEqual(languages, [None, None, "fil"])

    def test_payload_duplicate_removal(self):
        term = {
            "term_id": 12,
            "term": "example monitored phrase",
            "language": "ceb",
            "match_type": "phrase",
        }
        payload = self._payload(matched_terms=[term, dict(term)])
        self.assertEqual(payload["matched_terms"], [term])

    @patch("sender.http_client.requests.post")
    def test_legacy_send_alert_call_and_retry_path_remain_functional(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = {"id": 99, "severity": "high"}

        self.assertTrue(send_alert("high", 0.9, 2.0))
        posted = post.call_args.kwargs["json"]
        self.assertEqual(posted["language"], "unknown")
        self.assertIsNone(posted["language_confidence"])
        self.assertEqual(posted["matched_terms"], [])
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
