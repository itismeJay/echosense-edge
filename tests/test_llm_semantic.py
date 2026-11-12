"""Claude semantic classifier: API mocked, fusion table, threat floor, fallback."""

from datetime import datetime, timedelta, timezone
import inspect
import json
import queue
import threading
import time
from types import SimpleNamespace
import unittest

from detection.evidence import AcousticEvidence, LLMJudgement
from detection.fusion import (
    BANTER_REASON,
    DISTRESS_REASON,
    MEANING_REASON,
    REPEATED_REASON,
    THREAT_REVIEW_REASON,
    fuse_event_evidence,
)
from detection.llm_semantic import (
    DEFAULT_MODEL,
    OUTPUT_SCHEMA,
    ClaudeSemanticClient,
    RepetitionTracker,
    SemanticGate,
    SemanticInput,
    parse_judgement,
)
from detection.text_detector import TextDetector, TextDetectorWorker
from sender.incident_delivery import finalized_incident_send_arguments
from sender.payload_v2 import build_alert_payload_v2


WALL = datetime(2026, 11, 13, 1, 0, tzinfo=timezone.utc)
INSULT = "bobo ka"
THREAT = "patyon tika"
NO_WORD = "Ayaw siya pasakya sa among grupo"
GAME_THREAT = "patyon tika sa chess unya"
STORY_THREAT = "sa kwento ingon niya patyon tika"
LAUGH_THREAT = "patyon tika, hahaha"
REAL_THREATS = (
    "Patyon tika unya, hahaha",
    "Kulatahon tika pag-uli",
    "Susuntukin kita mamaya, joke lang",
    "I'll beat you up after class, haha",
    "Wait ka lang, bugbugon tika sa gawas",
)


def judgement(label="none", confidence=0.95, *, targeted=False, distress=False):
    return LLMJudgement(
        status="classified",
        label=label,
        targeted=targeted,
        victim_distress=distress,
        confidence=confidence,
        reason="test reason",
        model_id="claude-haiku-5-5",
    )


class FakeClient:
    """Returns a fixed judgement (or raises / sleeps) and records inputs."""

    def __init__(self, answer=None, *, delay=0.0, error=None):
        self.answer = answer
        self.delay = delay
        self.error = error
        self.inputs = []

    def classify(self, item):
        self.inputs.append(item)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        answer = self.answer(item) if callable(self.answer) else self.answer
        return answer


def gate_for(client, *, mode="all", timeout_s=2.0, logs=None):
    return SemanticGate(
        mode=mode,
        client=client,
        timeout_s=timeout_s,
        log=(logs.append if logs is not None else (lambda _line: None)),
    )


def analyze(text, answer=None, *, at=0.0, detector=None, mode="all", **gate_kwargs):
    detector = detector or TextDetector(
        semantic_gate=gate_for(FakeClient(answer), mode=mode, **gate_kwargs),
    )
    return detector.analyze_transcript(
        text,
        language="ceb",
        observed_at=WALL + timedelta(seconds=at),
        observed_monotonic=at,
    )


def fuse(*evidence, acoustic=()):
    return fuse_event_evidence(
        tuple(item for item in evidence if item.qualifies_to_open),
        tuple(acoustic),
        duration_seconds=1.0,
    )


def crying(at, *, score=0.6):
    return AcousticEvidence(
        interval_started_at=WALL + timedelta(seconds=at),
        interval_ended_at=WALL + timedelta(seconds=at + 0.975),
        monotonic_start=at,
        monotonic_end=at + 0.975,
        yamnet_ran=True,
        yamnet_label="Crying, sobbing",
        yamnet_score=score,
        yamnet_aggressive=False,
        tone_aggressive=False,
        rms=100.0,
        energy_variance=10.0,
        zero_crossing_rate=0.02,
        peak_to_average=2.0,
        duration_seconds=0.975,
        reasons=("normal",),
        qualifies_to_open=False,
    )


class ParseJudgementTests(unittest.TestCase):
    def valid(self, **changes):
        data = {
            "label": "banter", "targeted": True, "victim_distress": False,
            "confidence": 0.9, "reason": "Friends   joking\ntogether.",
        }
        data.update(changes)
        return json.dumps(data)

    def test_valid_answer_is_normalized(self):
        result = parse_judgement(self.valid(), model_id="m")
        self.assertTrue(result.ok)
        self.assertEqual(result.label, "banter")
        self.assertEqual(result.reason, "Friends joking together.")

    def test_invalid_answers_are_rejected(self):
        bad = [
            self.valid(label="bullying"),
            self.valid(confidence=1.5),
            self.valid(confidence=True),
            self.valid(targeted="yes"),
            json.dumps({"label": "none"}),
            "not json",
        ]
        for text in bad:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_judgement(text, model_id="m")


class ClaudeClientRequestTests(unittest.TestCase):
    def fake_sdk(self, text, stop_reason="end_turn"):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                stop_reason=stop_reason,
                model=DEFAULT_MODEL,
                content=[SimpleNamespace(type="text", text=text)],
                usage=SimpleNamespace(input_tokens=1200, output_tokens=60,
                                      cache_read_input_tokens=0,
                                      cache_creation_input_tokens=0),
            )

        return SimpleNamespace(messages=SimpleNamespace(create=create)), calls

    def test_request_uses_haiku_schema_and_all_inputs(self):
        answer = json.dumps({"label": "insult", "targeted": True,
                             "victim_distress": True, "confidence": 0.9,
                             "reason": "Name-calling."})
        sdk, calls = self.fake_sdk(answer)
        client = ClaudeSemanticClient(client=sdk)
        result = client.classify(SemanticInput(
            transcript="bogo ka", previous_turns=("a", "b"), language="ceb",
            matched_terms=("bogo",), emotion="angry", angry_tone=True,
            laughter=False, yamnet_crying_score=0.42,
        ))
        self.assertEqual(result.label, "insult")
        self.assertEqual(result.input_tokens, 1200)
        request = calls[0]
        self.assertEqual(request["model"], "claude-haiku-5-5")
        self.assertEqual(request["output_config"]["format"]["schema"], OUTPUT_SCHEMA)
        self.assertIn("Grade 6", request["system"][0]["text"])
        sent = json.loads(request["messages"][0]["content"])
        self.assertEqual(sent["previous_turns"], ["a", "b"])
        self.assertEqual(sent["matched_terms"], ["bogo"])
        self.assertEqual(sent["emotion"], "angry")
        self.assertTrue(sent["angry_tone"])
        self.assertEqual(sent["yamnet_crying_score"], 0.42)

    def test_refusal_becomes_fallback(self):
        sdk, _calls = self.fake_sdk("{}", stop_reason="refusal")
        gate = gate_for(ClaudeSemanticClient(client=sdk))
        result = gate.classify(SemanticInput(transcript="x"))
        self.assertEqual(result.status, "fallback")


class GateTests(unittest.TestCase):
    def test_timeout_falls_back_to_lexicon_only_and_logs(self):
        logs = []
        gate = gate_for(FakeClient(judgement("banter"), delay=0.5),
                        timeout_s=0.05, logs=logs)
        started = time.monotonic()
        result = gate.classify(SemanticInput(transcript="x"))
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual((result.status, result.fallback_reason), ("fallback", "timeout"))
        self.assertTrue(any("fallback=lexicon_only reason=timeout" in line for line in logs))
        gate.shutdown()

    def test_api_error_falls_back(self):
        gate = gate_for(FakeClient(error=ConnectionError("down")))
        result = gate.classify(SemanticInput(transcript="x"))
        self.assertEqual((result.status, result.fallback_reason), ("fallback", "ConnectionError"))

    def test_environment_defaults_and_key_never_logged(self):
        logs = []
        gate = SemanticGate.from_environment({}, log=logs.append)
        self.assertEqual(gate.mode, "all")
        self.assertFalse(gate.available)
        self.assertIn("reason=no_api_key", logs[-1])
        secret = "sk-ant-test-SECRET-123"
        logs.clear()
        gate = SemanticGate.from_environment(
            {"ANTHROPIC_API_KEY": secret, "ECHOSENSE_SEMANTIC_MODE": "on_match"},
            log=logs.append,
        )
        self.assertTrue(gate.available)
        self.assertEqual(gate.mode, "on_match")
        self.assertFalse(any(secret in line for line in logs))
        self.assertEqual(SemanticGate.from_environment(
            {"ECHOSENSE_SEMANTIC_MODE": "bogus"}, log=lambda _l: None).mode, "off")

    def test_modes_choose_which_utterances_are_sent(self):
        for mode, text, expected in (
            ("off", INSULT, 0), ("on_match", NO_WORD, 0),
            ("on_match", INSULT, 1), ("all", NO_WORD, 1),
        ):
            with self.subTest(mode=mode, text=text):
                client = FakeClient(judgement())
                analyze(text, detector=TextDetector(
                    semantic_gate=gate_for(client, mode=mode)))
                self.assertEqual(len(client.inputs), expected)

    def test_previous_two_turns_within_sixty_seconds(self):
        client = FakeClient(judgement())
        detector = TextDetector(semantic_gate=gate_for(client))
        for at, text in ((0, "una"), (30, "ikaduha"), (50, "ikatulo"), (100, "ikaupat")):
            analyze(text, at=at, detector=detector)
        self.assertEqual(client.inputs[2].previous_turns, ("una", "ikaduha"))
        self.assertEqual(client.inputs[3].previous_turns, ("ikatulo",))


class FusionTableTests(unittest.TestCase):
    """Expected (deliver, level, reason) for each lexicon x Claude case."""

    def test_fusion_table(self):
        table = [
            # text, Claude answer, delivered, level, reason expected
            (INSULT, None, True, "MEDIUM", None),
            (INSULT, judgement("banter", 0.85, targeted=True), True, "LOW", BANTER_REASON),
            (INSULT, judgement("friendly_slang", 0.9), True, "LOW", BANTER_REASON),
            (INSULT, judgement("banter", 0.7), True, "MEDIUM", None),
            (INSULT, judgement("insult", 0.9, targeted=True), True, "MEDIUM", None),
            (INSULT, judgement("insult", 0.9, targeted=True, distress=True), True, "HIGH", DISTRESS_REASON),
            (THREAT, judgement("banter", 0.99), True, "HIGH", None),
            (THREAT, judgement("quoting", 0.95), True, "HIGH", None),
            (THREAT, judgement("none", 0.95), True, "HIGH", None),
            (GAME_THREAT, judgement("banter", 0.95), True, "MEDIUM", THREAT_REVIEW_REASON),
            (STORY_THREAT, judgement("quoting", 0.95), True, "MEDIUM", THREAT_REVIEW_REASON),
            (GAME_THREAT, judgement("banter", 0.85), True, "HIGH", None),
            (GAME_THREAT, judgement("none", 0.99), True, "HIGH", None),
            (GAME_THREAT, judgement("friendly_slang", 0.99), True, "HIGH", None),
            (LAUGH_THREAT, judgement("banter", 0.99), True, "HIGH", None),
            (NO_WORD, judgement("exclusion", 0.9, targeted=True), True, "MEDIUM", MEANING_REASON),
            (NO_WORD, judgement("threat", 0.9, targeted=True), True, "HIGH", MEANING_REASON),
            (NO_WORD, judgement("insult", 0.8, targeted=True), False, None, None),
            (NO_WORD, judgement("insult", 0.95, targeted=False), False, None, None),
            (NO_WORD, judgement("banter", 0.95, targeted=True), False, None, None),
        ]
        for text, answer, delivered, level, reason in table:
            with self.subTest(text=text, answer=answer and (answer.label, answer.confidence)):
                mode = "off" if answer is None else "all"
                evidence = analyze(text, answer, mode=mode)
                if not delivered:
                    self.assertFalse(evidence.qualifies_to_open)
                    continue
                decision = fuse(evidence)
                self.assertTrue(decision.deliver)
                self.assertEqual(decision.level, level)
                if reason:
                    self.assertIn(reason, decision.reasons)

    def test_threat_never_below_medium(self):
        labels = ("insult", "threat", "exclusion", "shaming", "banter",
                  "friendly_slang", "profanity_exclamation", "quoting",
                  "self_directed", "none")
        for label in labels:
            for confidence in (0.0, 0.5, 0.8, 0.9, 1.0):
                with self.subTest(label=label, confidence=confidence):
                    decision = fuse(analyze(THREAT, judgement(label, confidence)))
                    self.assertTrue(decision.deliver)
                    self.assertIn(decision.level, ("MEDIUM", "HIGH"))
                    self.assertEqual(decision.level, "HIGH")
                    game = fuse(analyze(GAME_THREAT, judgement(label, confidence)))
                    self.assertIn(game.level, ("MEDIUM", "HIGH"))
                    if label not in ("banter", "quoting") or confidence < 0.9:
                        self.assertEqual(game.level, "HIGH")

    def test_real_threats_stay_high_whatever_the_classifier_says(self):
        answers = [None, judgement("banter", 0.99, targeted=True),
                   judgement("quoting", 0.99), judgement("friendly_slang", 0.99),
                   judgement("none", 0.99)]
        for text in REAL_THREATS:
            for answer in answers:
                with self.subTest(text=text, answer=answer and answer.label):
                    evidence = analyze(text, answer, mode="off" if answer is None else "all")
                    self.assertTrue(evidence.qualifies_to_open)
                    decision = fuse(evidence)
                    self.assertTrue(decision.deliver)
                    self.assertEqual(decision.level, "HIGH")

    def test_game_word_in_previous_turn_counts_laughter_alone_does_not(self):
        detector = TextDetector(semantic_gate=gate_for(FakeClient(judgement("banter", 0.95))))
        analyze("Tara laro ta ML", at=0, detector=detector)
        self.assertEqual(fuse(analyze(THREAT, at=20, detector=detector)).level, "MEDIUM")
        detector = TextDetector(semantic_gate=gate_for(FakeClient(judgement("banter", 0.95))))
        analyze("hahaha kataw-anan", at=0, detector=detector)
        self.assertEqual(fuse(analyze(LAUGH_THREAT, at=20, detector=detector)).level, "HIGH")

    def test_banter_is_never_dropped_and_flags_are_sent(self):
        decision = fuse(analyze(INSULT, judgement("banter", 0.9, targeted=True)))
        self.assertTrue(decision.deliver)
        self.assertIn("semantic_label:banter", decision.supporting_evidence)
        self.assertIn("semantic_confidence:0.90", decision.supporting_evidence)
        self.assertIn("semantic_reason:test reason", decision.supporting_evidence)
        self.assertIn("semantic_banter:true", decision.supporting_evidence)
        self.assertIn("semantic_repeated:false", decision.supporting_evidence)

    def test_angry_targeted_insult_keeps_high_despite_banter(self):
        from detection.fusion import ANGRY_BANTER_REASON, apply_semantic_rules
        item = analyze(INSULT, judgement("banter", 0.9, targeted=True))
        level, reasons, supporting, summary = apply_semantic_rules(
            "HIGH", item, (), {}, "x", angry_tone=True, targeted=True,
        )
        self.assertEqual(level, "HIGH")
        self.assertIn(ANGRY_BANTER_REASON, reasons)
        self.assertIn("semantic_banter:true", supporting)
        calm, calm_reasons, _s, _m = apply_semantic_rules(
            "HIGH", item, (), {}, "x", angry_tone=False, targeted=True,
        )
        self.assertEqual(calm, "LOW")
        self.assertNotIn(ANGRY_BANTER_REASON, calm_reasons)

    def test_crying_after_the_insult_raises_one_level(self):
        evidence = analyze(INSULT, judgement("insult", 0.9, targeted=True), at=10)
        after = fuse(evidence, acoustic=(crying(10.5),))
        before = fuse(evidence, acoustic=(crying(5.0),))
        self.assertEqual(after.level, "HIGH")
        self.assertIn(DISTRESS_REASON, after.reasons)
        self.assertIn("yamnet_crying_after_insult", after.supporting_evidence)
        self.assertEqual(before.level, "MEDIUM")

    def test_crying_raises_even_when_the_classifier_failed(self):
        evidence = analyze(INSULT, at=10, detector=TextDetector(
            semantic_gate=gate_for(FakeClient(error=TimeoutError()))))
        self.assertEqual(fuse(evidence, acoustic=(crying(10.2),)).level, "HIGH")


class FallbackTests(unittest.TestCase):
    def test_fallback_equals_lexicon_only(self):
        for text in (INSULT, THREAT, "yawa"):
            with self.subTest(text=text):
                baseline = fuse(analyze(text, mode="off"))
                failed = analyze(text, detector=TextDetector(
                    semantic_gate=gate_for(FakeClient(error=RuntimeError("boom")))))
                self.assertEqual(failed.llm.status, "fallback")
                decision = fuse(failed)
                self.assertEqual((decision.deliver, decision.level),
                                 (baseline.deliver, baseline.level))
                self.assertIn("semantic_fallback:lexicon_only:RuntimeError",
                              decision.supporting_evidence)

    def test_fallback_never_opens_by_meaning(self):
        evidence = analyze(NO_WORD, detector=TextDetector(
            semantic_gate=gate_for(FakeClient(error=RuntimeError()))))
        self.assertFalse(evidence.qualifies_to_open)


class RepetitionTests(unittest.TestCase):
    def test_third_similar_insult_within_ten_minutes_raises_one_level(self):
        detector = TextDetector(semantic_gate=gate_for(
            FakeClient(judgement("insult", 0.9, targeted=True))))
        results = [analyze(INSULT, at=at, detector=detector) for at in (0, 200, 500)]
        self.assertEqual([item.pattern_repeated for item in results], [False, False, True])
        decision = fuse(results[-1])
        self.assertEqual(decision.level, "HIGH")
        self.assertIn(REPEATED_REASON, decision.reasons)
        self.assertIn("semantic_repeated:true", decision.supporting_evidence)

    def test_spread_over_more_than_ten_minutes_is_not_repeated(self):
        detector = TextDetector()
        results = [analyze(INSULT, at=at, detector=detector, mode="off")
                   for at in (0, 400, 1100)]
        self.assertFalse(results[-1].pattern_repeated)

    def test_similar_means_shared_term_or_same_harmful_label(self):
        tracker = RepetitionTracker()
        self.assertEqual(tracker.observe({"bobo"}, None, 0), 1)
        self.assertEqual(tracker.observe({"tanga"}, None, 1), 1)
        self.assertEqual(tracker.observe((), "exclusion", 2), 1)
        self.assertEqual(tracker.observe({"bobo", "ka"}, None, 3), 2)
        self.assertEqual(tracker.observe((), "exclusion", 4), 2)


class PayloadTests(unittest.TestCase):
    def test_semantic_fields_travel_in_existing_evidence_fields(self):
        from detection.coordinator import EventCoordinator

        finalized = []
        clock = [0.0]
        coordinator = EventCoordinator(
            monotonic_clock=lambda: clock[0], finalizer=finalized.append,
        )
        evidence = analyze(NO_WORD, judgement("shaming", 0.9, targeted=True))
        coordinator.submit_text(evidence)
        clock[0] = 10.0
        coordinator.tick()
        arguments = finalized_incident_send_arguments(
            finalized[0], device_identifier="device-test", dashboard_test_mode=False,
        )
        accepted = inspect.signature(build_alert_payload_v2).parameters
        arguments.pop("duration")
        arguments["event_id"] = str(arguments["event_id"])
        payload = build_alert_payload_v2(
            **{key: value for key, value in arguments.items() if key in accepted}
        )
        self.assertEqual(payload["severity"].lower(), "medium")
        self.assertIn(MEANING_REASON, payload["severity_reasons"])
        supporting = payload["severity_evidence"]["supporting_evidence"]
        self.assertIn("semantic_label:shaming", supporting)
        self.assertIn("semantic_banter:false", supporting)
        self.assertFalse(payload["monitored_word_detected"])


class WorkerPipelineTests(unittest.TestCase):
    def test_calls_overlap_and_evidence_keeps_arrival_order(self):
        lock = threading.Lock()
        active = [0, 0]  # current, peak

        def answer(item):
            with lock:
                active[0] += 1
                active[1] = max(active[1], active[0])
            time.sleep(0.15 if item.transcript.endswith("1") else 0.05)
            with lock:
                active[0] -= 1
            return judgement()

        detector = TextDetector(semantic_gate=gate_for(FakeClient(answer)))
        results = queue.Queue()
        for index in range(4):
            results.put({"transcribed_text": f"pila ang answer {index}",
                         "quality_accepted": True, "language": "ceb"})
        seen = []

        def consume(timeout):
            try:
                return results.get(timeout=timeout)
            except queue.Empty:
                return None

        worker = TextDetectorWorker(detector, consume, seen.append)
        worker.start()
        deadline = time.monotonic() + 3
        while len(seen) < 4 and time.monotonic() < deadline:
            time.sleep(0.02)
        worker.stop()
        self.assertEqual([item.transcript[-1] for item in seen], ["0", "1", "2", "3"])
        self.assertGreaterEqual(active[1], 2)


if __name__ == "__main__":
    unittest.main()
