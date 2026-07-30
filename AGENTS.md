# Repository Guidelines

## Project Structure & Module Organization

EchoSense Edge is a Python service for Raspberry Pi audio-risk detection. `main.py`
is the runtime entry point. Audio capture and signal helpers live in `audio/`;
speech, term matching, YAMNet, and tone logic live in `model/`; decision gates
and severity rules live in `detection/`; and backend delivery, the SQLite outbox,
and operator commands live in `sender/`. Automated tests are under `tests/`.
Root-level `test_tone.py` is a standalone synthetic-audio check, while
`test_backend.py` contacts the configured backend. Operational notes are in
`docs/`, service overrides in `deploy/`, and device setup in `SCHOOL_SETUP.md`.

## Build, Test, and Development Commands

There is no compilation step. Create the documented `echosense-env` virtual
environment and install `requirements.txt`, then run commands from the repository
root:

```bash
echosense-env/bin/pip install -r requirements.txt
echosense-env/bin/python3 -m unittest discover -s tests -p 'test_*.py'
echosense-env/bin/python3 test_tone.py
echosense-env/bin/python3 main.py
```

The unit suite should not require microphone hardware or backend access.
`main.py` does require the device audio stack and ignored model assets.
Run `test_backend.py` only when intentionally sending a synthetic alert.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, `snake_case` for
modules/functions/variables, `PascalCase` for classes, and uppercase names for
constants. Keep imports grouped at the top and add concise docstrings where
behavior is non-obvious. No formatter or linter is configured, so keep changes
PEP 8-compatible and focused. Detection thresholds belong in
`detection/thresholds.py`.

## Testing Guidelines

Tests use the standard-library `unittest` framework, including mocks and
temporary directories. Name files `test_<feature>.py`, classes `*Tests`, and
methods `test_<behavior>`. Add regression coverage for changes to detection
gates, payloads, transcript privacy, audio alignment, or outbox state transitions.
Use deterministic synthetic inputs; do not depend on live audio or remote APIs.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commit-style subjects such as
`feat: synchronize edge audio evidence` and `fix: lower thresholds ...`. Use a
short imperative `feat:`, `fix:`, `test:`, or `docs:` subject. Pull requests
should explain the behavior change, list validation commands, link relevant
issues, and call out hardware/backend testing. Include screenshots or sanitized
logs only when they clarify operator-visible behavior.

## Security & Configuration

Do not commit `.env`, model files, SQLite outboxes, raw audio, exact transcripts,
or credentials. Preserve privacy-safe logging defaults. Configure deployments
through documented `ECHOSENSE_*` environment variables and never reset a
production outbox containing pending alerts.
