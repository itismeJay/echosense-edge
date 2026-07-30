# EchoSense Edge Severity Rules

The authoritative implementation is `detection/severity.py`. Severity describes
observable evidence in an unverified possible-aggression alert. It does not
confirm bullying, intent, guilt, immediate danger, or speaker identity.

| Severity | Required evidence | Supporting evidence | Suppression and exclusion rules | Example cases |
|---|---|---|---|---|
| `LOW` | A low-risk or uncategorized monitored term with no stronger contextual evidence | Limited text evidence | Narrow harmless-object rules may remove ambiguous terms before severity calculation. Laughter or excitement may suppress remaining limited evidence. | A configured low-risk term without a direct-target pattern |
| `MEDIUM` | A configured direct-harassment term, or a low-risk term with a direct-target pattern or aggressive acoustic support | Duration of at least 4 seconds may also set a MEDIUM floor | Narrow harmless-context rules still apply before calculation. Laughter may suppress MEDIUM evidence when no HIGH term is present. | A direct insult pattern; an audio-supported low-risk monitored term |
| `HIGH` | A self-harm directive, threat-like phrase, severe direct-harassment term, repeated MEDIUM evidence, or duration of at least 7 seconds | Laughter/excitement, direct-target patterns, repetition, and acoustic evidence remain recorded as context | Laughter, `haha`, `hehe`, excitement markers, or acoustic laughter cannot cancel HIGH text evidence. A transcript-quality rejection or narrow harmless-context exclusion still prevents unsupported evidence from reaching the calculator. | `kill yourself`; a configured threat-like phrase; repeated direct harassment |

## Explainability

Each decision exposes:

- `level`
- `reasons`
- matched `term_categories`
- `supporting_evidence`

Alert logs emit these fields without printing the complete transcript. Exact
transcript text remains unchanged in the existing transcript and alert fields.

## Compatibility

The edge uses `LOW`, `MEDIUM`, and `HIGH` internally. The current production API
uses lowercase values, so `sender/http_client.py` converts severity to lowercase
only while building the outgoing payload. No other layer should convert labels.

## Known limitations

- Severity does not establish intent or confirm bullying.
- The system does not identify speakers, a bully, or a victim.
- Text and acoustic evidence can be affected by transcription and classification
  errors.
- Laughter may coexist with serious language; it is retained as context rather
  than treated as proof that the phrase was harmless.
- Controlled rules cannot cover every classroom expression or language variant.
- Human review is required for every alert.
