"""The school config from /devices/me/config: words, turned-off words, sensitivity, cache."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from detection import school_policy
from detection.fusion import fuse_event_evidence
from detection.soundalike import _allowed_mismatches
from detection.text_detector import TextDetector
from model import monitored_terms, school_config
from model.blacklist import check_transcript


def config_body(*, terms=(), disabled=(), sensitivity="normal", version="v1"):
    return {
        "schema_version": 1,
        "device_code": "test-device",
        "school_id": "00000000-0000-0000-0000-000000000001",
        "sensitivity": sensitivity,
        "terms": list(terms),
        "disabled_terms": list(disabled),
        "config_version": version,
        "generated_at": "2026-11-05T00:00:00Z",
    }


def school_word(text, term_id=7, language="ceb"):
    return {
        "text": text,
        "language": language,
        "severity_weight": 0.6,
        "source": "school",
        "term_id": term_id,
    }


def global_word(text, term_id, language="ceb"):
    return {
        "text": text,
        "language": language,
        "severity_weight": 0.8,
        "source": "global",
        "term_id": term_id,
    }


class Response:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.urls = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class SchoolConfigTestCase(unittest.TestCase):
    def setUp(self):
        self._saved_terms = monitored_terms.get_backend_terms()
        self._saved_version = school_config._applied_version
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name) / "school_config.json"
        patcher = mock.patch.dict(
            "os.environ", {"ECHOSENSE_SCHOOL_CONFIG_PATH": str(self.cache)}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        school_config._applied_version = None
        school_policy.reset_policy()
        monitored_terms.set_backend_terms(())

    def tearDown(self):
        monitored_terms.set_backend_terms(self._saved_terms)
        school_config._applied_version = self._saved_version
        school_policy.reset_policy()
        self._tmp.cleanup()


class SchoolWordsTest(SchoolConfigTestCase):
    def test_a_school_word_alerts_and_is_sent_as_text_for_the_backend_to_link(self):
        school_config.apply_config(config_body(terms=[school_word("sekreto nga pulong")]))

        result = check_transcript("sekreto nga pulong ka")

        self.assertTrue(result["has_profanity"])
        self.assertIn("sekreto nga pulong", result["detected_words"])
        [evidence] = [
            item for item in result["matched_terms"] if item["term"] == "sekreto nga pulong"
        ]
        # No term_id: a school word's id must never be mistaken for a global id.
        self.assertNotIn("term_id", evidence)
        self.assertEqual(evidence["language"], "ceb")

    def test_global_words_keep_their_term_ids(self):
        school_config.apply_config(config_body(terms=[global_word("bogo", 41)]))

        result = check_transcript("bogo ka")

        self.assertIn({"term": "bogo", "language": "ceb", "match_type": "word",
                       "term_id": 41}, result["matched_terms"])


class TurnedOffWordsTest(SchoolConfigTestCase):
    def test_a_turned_off_built_in_phrase_does_not_alert(self):
        self.assertTrue(check_transcript("tambok ka")["has_profanity"])

        school_config.apply_config(config_body(disabled=["tambok ka"]))
        result = check_transcript("tambok ka")

        # "tambok" alone is also a built-in word, but it is part of the turned-off phrase.
        self.assertFalse(result["has_profanity"])
        self.assertEqual(result["detected_words"], [])
        self.assertEqual(result["matched_terms"], [])

    def test_turning_off_one_word_keeps_the_others(self):
        school_config.apply_config(config_body(disabled=["tambok ka"]))

        result = check_transcript("bogo ka")

        self.assertTrue(result["has_profanity"])
        self.assertIn("bogo ka", result["detected_words"])

    def test_a_turned_off_word_does_not_open_an_incident(self):
        school_config.apply_config(config_body(disabled=["tambok ka"]))

        evidence = TextDetector().analyze_transcript("tambok ka")

        self.assertFalse(evidence.qualifies_to_open)

    def test_turned_off_words_match_after_normalization(self):
        school_config.apply_config(config_body(disabled=["  Tambok KA "]))

        self.assertFalse(check_transcript("tambok ka")["has_profanity"])


def descriptive_untargeted_decision():
    evidence = TextDetector().analyze_transcript("pangit kaayo")
    assert evidence.qualifies_to_open, evidence
    return fuse_event_evidence((evidence,), (), duration_seconds=1.0)


class SensitivityTest(SchoolConfigTestCase):
    def test_normal_holds_back_an_untargeted_descriptive_word(self):
        school_config.apply_config(config_body(sensitivity="normal"))

        decision = descriptive_untargeted_decision()

        self.assertFalse(decision.deliver)
        self.assertIn("descriptive_untargeted:never_sent", decision.reasons)

    def test_strict_sends_an_untargeted_descriptive_word(self):
        school_config.apply_config(config_body(sensitivity="strict"))

        decision = descriptive_untargeted_decision()

        self.assertTrue(decision.deliver)
        self.assertIn(
            "strict_sensitivity:descriptive_untargeted_sent", decision.supporting_evidence
        )

    def test_strict_alerts_on_a_bare_ambiguous_word(self):
        self.assertFalse(check_transcript("pangit")["has_profanity"])

        school_config.apply_config(config_body(sensitivity="strict"))

        self.assertTrue(check_transcript("pangit")["has_profanity"])

    def test_strict_still_keeps_harmless_and_negated_uses_quiet(self):
        school_config.apply_config(config_body(sensitivity="strict"))

        self.assertFalse(check_transcript("pangit ang weather")["has_profanity"])
        self.assertFalse(check_transcript("dili ka bogo")["has_profanity"])

    def test_strict_allows_one_more_differing_sound_on_longer_single_words(self):
        def allowed(single_word):
            return [_allowed_mismatches(n, single_word=single_word) for n in (4, 6, 7, 9)]

        self.assertEqual(allowed(True), [1, 1, 1, 2])

        school_policy.set_policy(school_policy.STRICT)

        self.assertEqual(allowed(True), [1, 1, 2, 2])
        # Phrases keep the normal rule.
        self.assertEqual(allowed(False), [1, 1, 1, 2])

    def test_strict_never_turns_dili_ka_into_an_insult(self):
        school_config.apply_config(config_body(sensitivity="strict"))

        result = check_transcript("dili ka")

        self.assertFalse(result["has_profanity"])
        self.assertEqual(result["corrections"], [])

    def test_unknown_sensitivity_falls_back_to_normal(self):
        school_config.apply_config(config_body(sensitivity="paranoid"))

        self.assertEqual(school_policy.sensitivity(), school_policy.NORMAL)


class SyncTest(SchoolConfigTestCase):
    def test_sync_applies_the_config_and_caches_it(self):
        body = config_body(terms=[school_word("sekreto nga pulong")], disabled=["tambok ka"],
                           sensitivity="strict")
        session = Session(Response(200, body))

        outcome = school_config.sync_school_config(session=session)

        self.assertEqual(outcome, school_config.APPLIED)
        self.assertTrue(session.urls[0].endswith("/devices/me/config"))
        self.assertTrue(school_policy.is_strict())
        self.assertEqual(school_policy.disabled_terms(), frozenset({"tambok ka"}))
        self.assertEqual(school_config.load_cache(self.cache), body)
        self.assertEqual(self.cache.stat().st_mode & 0o777, 0o600)

    def test_an_unchanged_version_is_not_reapplied(self):
        body = config_body()
        school_config.sync_school_config(session=Session(Response(200, body)))

        outcome = school_config.sync_school_config(session=Session(Response(200, body)))

        self.assertEqual(outcome, school_config.UNCHANGED)

    def test_offline_start_uses_the_cached_config(self):
        body = config_body(disabled=["tambok ka"], sensitivity="strict")
        school_config.save_cache(body, self.cache)

        outcome = school_config.sync_school_config(
            session=Session(ConnectionError("offline"))
        )

        self.assertEqual(outcome, school_config.CACHED)
        self.assertTrue(school_policy.is_strict())
        self.assertFalse(check_transcript("tambok ka")["has_profanity"])

    def test_a_failed_refresh_keeps_the_current_config(self):
        school_config.sync_school_config(
            session=Session(Response(200, config_body(sensitivity="strict")))
        )

        outcome = school_config.sync_school_config(session=Session(Response(503)))

        self.assertEqual(outcome, school_config.FAILED)
        self.assertTrue(school_policy.is_strict())

    def test_another_devices_cache_is_ignored(self):
        self.cache.write_text(json.dumps({
            "api_url": school_config.API_URL,
            "device_id": "someone-else",
            "config": config_body(sensitivity="strict"),
        }))

        outcome = school_config.sync_school_config(
            session=Session(ConnectionError("offline"))
        )

        self.assertEqual(outcome, school_config.FAILED)
        self.assertFalse(school_policy.is_strict())

    def test_an_older_backend_falls_back_to_the_dictionary(self):
        school_policy.set_policy(school_policy.STRICT, {"tambok ka"})
        session = Session(
            Response(404),
            Response(200, [{"term_id": 3, "slur_text": "bogo", "language": "ceb"}]),
        )

        outcome = school_config.sync_school_config(session=session)

        self.assertEqual(outcome, school_config.DICTIONARY)
        self.assertTrue(session.urls[1].endswith("/dictionary/"))
        self.assertFalse(school_policy.is_strict())
        self.assertEqual(school_policy.disabled_terms(), frozenset())
        self.assertEqual([t.term for t in monitored_terms.get_backend_terms()], ["bogo"])

    def test_an_invalid_body_is_not_applied(self):
        outcome = school_config.sync_school_config(
            session=Session(Response(200, {"schema_version": 2}))
        )

        self.assertEqual(outcome, school_config.FAILED)
        self.assertIsNone(school_config.applied_version())

    def test_a_rejected_alert_resyncs_the_school_config_not_the_bare_dictionary(self):
        from sender import delivery

        with mock.patch.object(school_config, "resync_terms", return_value=True) as resync:
            self.assertTrue(delivery._default_resync())
        resync.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
