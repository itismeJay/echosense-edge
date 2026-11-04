"""Decode gate: a monitored term relaxes only the avg_logprob check."""

import unittest

from detection.false_alarm_filters import (
    DecodeMetrics,
    FilterConfig,
    TERM_AVG_LOGPROB_MIN_DEFAULT,
    decode_gate,
)

DEFAULTS = FilterConfig()


def metrics(*, no_speech=0.05, logprob=-0.3, compression=1.2):
    return DecodeMetrics(
        no_speech_prob=no_speech,
        avg_logprob=logprob,
        compression_ratio=compression,
    )


class TermAwareDecodeGateTests(unittest.TestCase):
    def test_default_term_floor(self):
        self.assertEqual(TERM_AVG_LOGPROB_MIN_DEFAULT, -1.5)
        self.assertEqual(DEFAULTS.term_avg_logprob_min, -1.5)

    def test_without_a_term_the_strict_floor_applies(self):
        self.assertEqual(
            decode_gate(metrics(logprob=-1.12), DEFAULTS), "avg_logprob<-1"
        )
        self.assertEqual(
            decode_gate(metrics(logprob=-1.12), DEFAULTS, ["  "]), "avg_logprob<-1"
        )

    def test_with_a_term_low_avg_logprob_alone_keeps_the_utterance(self):
        # Measured on the test set: "Boongi ka" -> bungi ka at -1.121.
        self.assertIsNone(
            decode_gate(metrics(logprob=-1.121), DEFAULTS, ["bungi ka"])
        )

    def test_with_a_term_the_term_floor_still_applies(self):
        self.assertEqual(
            decode_gate(metrics(logprob=-1.6), DEFAULTS, ["bungi ka"]),
            "avg_logprob<-1.5",
        )

    def test_hallucination_signs_still_drop_an_utterance_with_a_term(self):
        self.assertEqual(
            decode_gate(metrics(no_speech=0.61), DEFAULTS, ["bobo"]),
            "no_speech_prob>0.6",
        )
        self.assertEqual(
            decode_gate(metrics(compression=2.41), DEFAULTS, ["bobo"]),
            "compression_ratio>2.4",
        )

    def test_term_floor_can_be_switched_off(self):
        config = FilterConfig(term_avg_logprob_min=None)
        self.assertIsNone(decode_gate(metrics(logprob=-3.0), config, ["bobo"]))
        self.assertEqual(
            decode_gate(metrics(logprob=-3.0), config), "avg_logprob<-1"
        )

    def test_environment(self):
        self.assertEqual(
            FilterConfig.from_environment({}).term_avg_logprob_min, -1.5
        )
        self.assertEqual(
            FilterConfig.from_environment(
                {"ECHOSENSE_FILTER_TERM_AVG_LOGPROB_MIN": "-2"}
            ).term_avg_logprob_min,
            -2.0,
        )
        self.assertIsNone(
            FilterConfig.from_environment(
                {"ECHOSENSE_FILTER_TERM_AVG_LOGPROB_MIN": "off"}
            ).term_avg_logprob_min
        )
        self.assertEqual(
            FilterConfig.from_environment(
                {"ECHOSENSE_FILTER_TERM_AVG_LOGPROB_MIN": "junk"}
            ).term_avg_logprob_min,
            -1.5,
        )


if __name__ == "__main__":
    unittest.main()
