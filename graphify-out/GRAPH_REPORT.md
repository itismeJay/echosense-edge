# Graph Report - echosense-edge  (2026-11-10)

## Corpus Check
- 141 files · ~204,240 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: .conf 2, (none) 1, .service 1)

## Summary
- 2747 nodes · 6547 edges · 161 communities (136 shown, 25 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 552 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7c6d023e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evaluate_testset.py
- severity.py
- AcousticDetector
- school_config.py
- main.py
- http_client.py
- blacklist.py
- transcript_quality.py
- AlertOutbox
- DeliveryResult
- numpy
- NetworkManager
- credentials.py
- TextDetector
- EventCoordinator
- fuse_event_evidence
- test_wifi_setup.py
- semantic.py
- replace_backend_terms
- soundalike.py
- .from_float32_samples
- LEDIndicator
- check_transcript
- portal.py
- config.py
- AggressionDetector
- whisper_stt.py
- payload_v2.py
- CoordinatorHarness
- .run_incident
- test_school_config.py
- sent
- pathlib
- acoustic_evidence
- OutboxDeliveryWorker
- test_qr_setup.py
- RollingAudioBuffer
- activation.py
- test_semantic_upgrade.py
- .process
- exclusion
- .run_utterance
- test_false_positive_provenance.py
- FakeResponse
- AudioFrame
- NoiseReducer
- TranscriptVisibilityTests
- runtime.py
- DeliverySettings
- run_yamnet_float32
- WifiSetupService
- provision_box.py
- dashboard_test_mode.py
- decode_gate
- LatencyTracker
- evaluate_semantic_upgrade.py
- roots
- test_stt_settings.py
- MultilingualPhraseDetectionTests
- RealtimeSTTAudioEventAdapter
- schema_v2_payload
- test_severity_and_laughter.py
- SemanticAnalyzer
- filter_terms_by_probability
- roots
- networkmanager.py
- deliver_alert_payload
- test_edge_env.py
- text_context.py
- pobre
- load_identity
- ControllerRecoveryTests
- test_dashboard_test_mode.py
- test_accuracy_context.py
- PyAudioSharedSource
- patterns
- RestrictedWhisperExecutor
- stt_model_version
- .start
- false_alarm_filters.py
- ScriptedClassifier
- write_identity
- EchoSense School Setup Guide
- categories
- _run_transcription_cycle
- EdgeRuntime
- test_false_alarm_filters.py
- SemanticClassifier
- Persistent Offline Alert Outbox
- language_restricted_engine.py
- RealtimeSTTRecordingLimiter
- AudioAlignmentTests
- EngineTests
- ActivationTests
- media_gate
- Raspberry Pi 5 Wi-Fi Provisioning
- RejectionResyncTests
- test_tone.py
- validate_dashboard_test_configuration
- choose_language
- run
- Repository Guidelines
- capture.py
- CLAUDE.md
- shadow_log.py
- FilterConfig
- atay
- roots
- pangit
- piste
- backend_base_url_from_environment
- parse_boolean
- Multilingual sentence-model research
- animal
- baho
- bogo
- buang
- buktot
- bungi
- buwisit
- hugaw
- itom
- kutohon
- leche
- linti
- maot
- niwang
- pandak
- tambok
- yawa
- amaw
- ampon
- bati
- bayot
- bisaya
- bobo
- dakog ilong
- gaga
- gago
- punyeta
- PortalRouteTests
- FakeTensor
- Raspberry Pi Device Authentication
- EchoSense Edge Severity Rules
- pulubi
- sumbagon
- outbox_admin.py
- Dashboard Test Mode
- False-alert validation
- bunalan
- latigohon
- lumpagon
- sagpaon
- sipaon
- tadyakan
- roots
- RestrictedWhisperStartupError

## God Nodes (most connected - your core abstractions)
1. `check_transcript()` - 84 edges
2. `TextDetector` - 59 edges
3. `AlertOutbox` - 55 edges
4. `EventCoordinator` - 52 edges
5. `fuse_event_evidence()` - 43 edges
6. `OutboxDeliveryWorker` - 39 edges
7. `DetectionRuntimeConfig` - 37 edges
8. `AcousticDetector` - 35 edges
9. `AggressionDetector` - 34 edges
10. `calculate_severity()` - 34 edges

## Surprising Connections (you probably didn't know these)
- `Purpose and architecture` --references--> `AudioEvent`  [INFERRED]
  docs/OFFLINE_OUTBOX.md → audio/audio_event.py
- `Schema-v2 and privacy boundary` --references--> `AudioEvent`  [INFERRED]
  docs/OFFLINE_OUTBOX.md → audio/audio_event.py
- `Current state` --references--> `TransformerNliClassifier`  [INFERRED]
  docs/EXPERIMENTAL_SEMANTICS.md → model/semantic_nli.py
- `Known landmines` --references--> `load_yamnet_bundle()`  [INFERRED]
  CLAUDE.md → model/yamnet_infer.py
- `AFTER detection architecture` --references--> `AudioEvent`  [INFERRED]
  docs/after_detection_architecture.md → audio/audio_event.py

## Import Cycles
- None detected.

## Communities (161 total, 25 thin omitted)

### Community 0 - "evaluate_testset.py"
Cohesion: 0.05
Nodes (23): _clean(), _expand(), add(), Lexicon, load_lexicon(), Phrase, _role_for(), Root (+15 more)

### Community 1 - "severity.py"
Cohesion: 0.06
Nodes (26): _targeted_utterances(), calculate_severity(), decision(), contains_high_severity_term(), _context_downgrade(), has_direct_target_evidence(), is_descriptive_only(), _is_descriptive_term() (+18 more)

### Community 2 - "AcousticDetector"
Cohesion: 0.08
Nodes (14): AcousticDetector, NoiseFloorTracker, AcousticEvidence, is_aggressive_sound(), load_class_names(), load_yamnet(), load_yamnet_bundle(), normalize_yamnet_label() (+6 more)

### Community 3 - "school_config.py"
Cohesion: 0.08
Nodes (20): disabled_terms(), is_strict(), reset_policy(), sensitivity(), set_policy(), set_backend_terms(), applied_version(), _apply_cache_if_nothing_applied() (+12 more)

### Community 4 - "main.py"
Cohesion: 0.09
Nodes (28): get_ip(), get_mac(), get_network_info(), get_ssid(), main(), _log_print(), _network_status_thread(), print_network_banner() (+20 more)

### Community 5 - "http_client.py"
Cohesion: 0.09
Nodes (16): get_alert_delivery_service(), get_outbox_health(), start_alert_delivery(), stop_alert_delivery(), device_auth_headers(), flush_logs(), _heartbeat_loop(), rejection_detail() (+8 more)

### Community 6 - "blacklist.py"
Cohesion: 0.09
Nodes (24): phrase_matches(), clean_text(), contains_blacklisted_word(), _correction_terms(), _correction_terms_for(), drop_turned_off_terms(), _exact_match(), _find_hits() (+16 more)

### Community 7 - "transcript_quality.py"
Cohesion: 0.08
Nodes (16): assess_transcript_quality(), _has_repeated_phrase_pattern(), _is_internal_apostrophe(), _is_word_character(), _longest_repeated_run(), normalize_transcript(), _normalized_characters(), phrase_ends_segment() (+8 more)

### Community 8 - "AlertOutbox"
Cohesion: 0.09
Nodes (11): AlertOutbox, _contains_raw_audio(), EnqueueResult, OutboxError, OutboxRecord, OutboxSchemaError, OutboxSerializationError, OutboxStorageError (+3 more)

### Community 9 - "DeliveryResult"
Cohesion: 0.12
Nodes (8): DeliveryResult, event_id_for(), AlertOutboxStorageTests, MutableClock, OutboxTestCase, OutboxWorkerTests, slow_delivery(), test_payload()

### Community 10 - "numpy"
Cohesion: 0.10
Nodes (9): AudioEvent, float32_samples_to_int16(), is_voice_present(), get_waveform_snapshot(), get_waveform_snapshot_float32(), analyze_tone(), analyze_tone_float32(), classify_emotion() (+1 more)

### Community 11 - "NetworkManager"
Cohesion: 0.10
Nodes (8): CommandResult, NetworkManager, SavedNetwork, _split_escaped(), NetworkManagerTests, runner(), runner(), RecordingRunner

### Community 12 - "credentials.py"
Cohesion: 0.09
Nodes (15): bootstrap(), build_parser(), _mac_address(), _machine_id(), main(), preflight(), create_setup_credentials(), device_suffix() (+7 more)

### Community 13 - "TextDetector"
Cohesion: 0.09
Nodes (12): _aware(), DetectedTermProvenance, _finite(), MatchedTermEvidence, TextEvidence, TranscriptCorrectionEvidence, _record_latency(), TextDetector (+4 more)

### Community 14 - "EventCoordinator"
Cohesion: 0.11
Nodes (5): ActiveIncident, ClosedTextIncident, EventCoordinator, _OutboxService, TranscriptPrivacyAfterTests

### Community 15 - "fuse_event_evidence"
Cohesion: 0.10
Nodes (12): CoordinatorSubmission, FinalizedIncident, _covered_seconds(), EventFusionDecision, fuse_event_evidence(), _fuse_text(), is_time_linked(), linked_acoustic() (+4 more)

### Community 16 - "test_wifi_setup.py"
Cohesion: 0.12
Nodes (11): network_fingerprint(), ProvisioningBusyError, ProvisioningController, connect(), connect_saved(), validate_profile_uuid(), validate_ssid(), validate_wifi_password() (+3 more)

### Community 17 - "semantic.py"
Cohesion: 0.11
Nodes (16): ConversationTurn, SemanticPrediction, SemanticRequest, create_classifier(), _entailment_index(), _load_transformer_components(), _normalized_label(), _premise() (+8 more)

### Community 18 - "replace_backend_terms"
Cohesion: 0.07
Nodes (10): replace_backend_terms(), start_dictionary_sync(), _sync_loop(), sync_backend_dictionary(), AttachBackendTermIdsTests, get(), get(), SyncResponse (+2 more)

### Community 19 - "soundalike.py"
Cohesion: 0.10
Nodes (19): _allowed_mismatches(), _changes_blocked_word(), _changes_common_word(), common_words(), correct_transcript(), Correction, CorrectionResult, _edit_distance_within() (+11 more)

### Community 20 - ".from_float32_samples"
Cohesion: 0.11
Nodes (3): BoundedResultQueue, AudioEventTests, FinalizedSttPcmBoundTests

### Community 21 - "LEDIndicator"
Cohesion: 0.09
Nodes (4): LEDIndicator, _read_active_trigger(), _read_on_value(), _write_led()

### Community 22 - "check_transcript"
Cohesion: 0.09
Nodes (6): categories_for_terms(), check_transcript(), correct_for_matching(), word_matches(), AfterTextCharacterizationTests, SoundalikeTests

### Community 23 - "portal.py"
Cohesion: 0.12
Nodes (7): _cookie_token(), _page(), PortalHandler, PortalSecurity, PortalServer, Session, PortalSecurityTests

### Community 24 - "config.py"
Cohesion: 0.11
Nodes (7): detection_runtime_config_from_environment(), DetectionRuntimeConfig, project_path(), semantic_runtime_config_from_environment(), _source_float(), _source_int(), AfterConfigurationTests

### Community 25 - "AggressionDetector"
Cohesion: 0.20
Nodes (8): AggressionDetector, process_transcription_result(), _alert_for(), _event(), _stt_result(), TextPipelineHardeningTests, tone(), _tone_result()

### Community 26 - "whisper_stt.py"
Cohesion: 0.11
Nodes (16): restrict_to_terms(), _apply_term_filters(), consume_transcription_result(), _empty_result(), get_model(), get_recorder(), _is_supported_transcription(), _log_decode_metrics() (+8 more)

### Community 27 - "payload_v2.py"
Cohesion: 0.18
Nodes (17): _acoustic_evidence(), AlertPayloadValidationError, _bounded_strings(), build_alert_payload_v2(), _clean_matched_terms(), _clean_terms(), _clean_waveform(), _evidence_role() (+9 more)

### Community 28 - "CoordinatorHarness"
Cohesion: 0.23
Nodes (7): CoordinatorHarness, IncidentWindowTests, text(), TimeLinkedCorroborationTests, windows(), DuplicateGuardTests, Speech

### Community 29 - ".run_incident"
Cohesion: 0.13
Nodes (8): enqueue_finalized_incident(), finalized_incident_send_arguments(), _language_for(), _strongest_acoustic(), _transcript_for(), IncidentDeliveryAfterTests, _OutboxService, _Response

### Community 30 - "test_school_config.py"
Cohesion: 0.15
Nodes (10): apply_config(), config_body(), descriptive_untargeted_decision(), global_word(), Response, school_word(), SchoolConfigTestCase, SchoolWordsTest (+2 more)

### Community 31 - "sent"
Cohesion: 0.10
Nodes (7): BisayaPriorityTests, Grade6BisayaTests, Grade6RecordingTests, ResplitTests, DescriptiveWordPolicyTests, MishearingTests, sent()

### Community 32 - "pathlib"
Cohesion: 0.12
Nodes (7): LiveValidationTests, configure_isolated_environment(), disable_realtimestt_log_file(), install_text_diagnostics(), main(), parse_args(), run_validation()

### Community 33 - "acoustic_evidence"
Cohesion: 0.22
Nodes (3): acoustic_evidence(), EventCoordinatorAfterTests, text_evidence()

### Community 35 - "test_qr_setup.py"
Cohesion: 0.15
Nodes (9): resolve_device_credentials(), load_device_key(), _read_private_json(), resolve_credentials(), ResolvedCredentials, save_device_key(), _tighten(), write_private_json() (+1 more)

### Community 36 - "RollingAudioBuffer"
Cohesion: 0.18
Nodes (5): AudioFormat, FakeAudioSource, RollingAudioBuffer, select_input_device(), SharedAudioAfterTests

### Community 37 - "activation.py"
Cohesion: 0.12
Nodes (8): activate_if_needed(), activate(), activate_until_ready(), _retry_after(), speak_command(), Speaker, IdentityError, SpeakerTests

### Community 38 - "test_semantic_upgrade.py"
Cohesion: 0.15
Nodes (8): SemanticRuntimeConfig, binary_metrics(), EvaluationObservation, grouped_comparison(), _percentile(), semantic_label_metrics(), is_laughter_sound(), SemanticUpgradeTests

### Community 39 - ".process"
Cohesion: 0.12
Nodes (7): _distinct_hits(), get_required_duration(), _laughter_suppresses(), _log_severity(), _severity_for_result(), ContextGate, SeverityDecision

### Community 40 - "exclusion"
Cohesion: 0.09
Nodes (23): exclusion, everyday, extra, kind, meaning, priority, everyday, extra (+15 more)

### Community 41 - ".run_utterance"
Cohesion: 0.17
Nodes (4): clear_transcription_results(), audio_event(), metadata(), PipelineTests

### Community 42 - "test_false_positive_provenance.py"
Cohesion: 0.14
Nodes (6): evidence_for(), HarmlessReferenceTests, IyotRegressionTests, LabeledTranscriptFixtureTests, MatchProvenanceTests, text_only_delivers()

### Community 43 - "FakeResponse"
Cohesion: 0.19
Nodes (3): DeliveryClassificationTests, FakeResponse, RejectionClassificationTests

### Community 44 - "AudioFrame"
Cohesion: 0.14
Nodes (3): AudioFrame, BoundedFrameQueue, AcousticDetectorWorker

### Community 45 - "NoiseReducer"
Cohesion: 0.12
Nodes (4): build_initial_prompt(), NoiseReducer, RealtimeSTTFeedWorker, WhisperFrontEndTests

### Community 48 - "DeliverySettings"
Cohesion: 0.14
Nodes (5): device_credentials_from_environment(), DeliverySettings, AuthenticatedOutboxTests, DeviceConfigurationTests, MutableClock

### Community 49 - "run_yamnet_float32"
Cohesion: 0.11
Nodes (7): run_yamnet(), run_yamnet_float32(), _run_yamnet_float32(), run_yamnet_scan(), scan_audio_float32(), _scan_windows(), _FakeInterpreter

### Community 51 - "provision_box.py"
Cohesion: 0.21
Nodes (13): BoxIdentity, _detail(), _font(), login(), main(), parse_args(), provision(), ProvisionError (+5 more)

### Community 52 - "dashboard_test_mode.py"
Cohesion: 0.18
Nodes (10): DashboardTestRunnerTests, configure_environment(), _env_values(), _load_device_credentials(), main(), _origin(), parse_args(), _reviewer_token() (+2 more)

### Community 53 - "decode_gate"
Cohesion: 0.20
Nodes (4): decode_gate(), metrics(), TermAwareDecodeGateTests, DecodeGateTests

### Community 55 - "evaluate_semantic_upgrade.py"
Cohesion: 0.20
Nodes (7): SemanticEvaluationToolTests, _adapter(), _fused_alert(), main(), _peak_rss_mb(), _read_cases(), _run_case()

### Community 56 - "roots"
Cohesion: 0.11
Nodes (18): threats, kind, meaning, priority, everyday, extra, kind, meaning (+10 more)

### Community 57 - "test_stt_settings.py"
Cohesion: 0.18
Nodes (5): apply_gain(), decode_restricted(), forced_language(), DecodeSettingsTests, FakeModel

### Community 58 - "MultilingualPhraseDetectionTests"
Cohesion: 0.17
Nodes (3): classify_transcript_language(), _real_confidence(), MultilingualPhraseDetectionTests

### Community 60 - "schema_v2_payload"
Cohesion: 0.18
Nodes (5): check_backend_connection(), send_alert(), main(), schema_v2_payload(), AlertPayloadV2Tests

### Community 61 - "test_severity_and_laughter.py"
Cohesion: 0.21
Nodes (4): _event(), _neutral_tone(), _result(), SeverityAndLaughterRegressionTests

### Community 63 - "filter_terms_by_probability"
Cohesion: 0.23
Nodes (6): filter_terms_by_probability(), normalize_words(), _similar(), term_probability(), metrics(), WordProbabilityTests

### Community 64 - "roots"
Cohesion: 0.12
Nodes (17): everyday, extra, kind, meaning, priority, roots, abnoy, ulaga (+9 more)

### Community 65 - "networkmanager.py"
Cohesion: 0.16
Nodes (3): ConnectionStatus, WifiNetwork, FakeManager

### Community 66 - "deliver_alert_payload"
Cohesion: 0.24
Nodes (7): deliver_alert_payload(), AuthenticatedHttpDeliveryTests, post(), post(), post(), FakeResponse, test_payload()

### Community 67 - "test_edge_env.py"
Cohesion: 0.18
Nodes (6): detect_platform(), load_edge_env(), parse_env_file(), DetectPlatformTests, LoadEdgeEnvTests, ParseEnvFileTests

### Community 68 - "text_context.py"
Cohesion: 0.20
Nodes (9): apply_harmless_context_rules(), _contains_sequence(), _negated(), _negation_reason(), _occurrences(), SuppressedTerm, _suppression_reason(), _term_reference_reason() (+1 more)

### Community 69 - "pobre"
Cohesion: 0.12
Nodes (16): poverty_family, everyday, extra, kind, meaning, priority, everyday, exclamative (+8 more)

### Community 70 - "load_identity"
Cohesion: 0.17
Nodes (4): load_identity(), IdentityFileTests, _mode(), ProvisionToolTests

### Community 72 - "test_dashboard_test_mode.py"
Cohesion: 0.18
Nodes (4): DashboardVisibilityVerifier, DashboardVisibilityVerifierTests, get(), _Response

### Community 73 - "test_accuracy_context.py"
Cohesion: 0.16
Nodes (3): AcousticOnlyPolicyTests, tone_window(), FakeClock

### Community 75 - "patterns"
Cohesion: 0.13
Nodes (14): language, patterns, adjective, exclamative, noun, swear, swear_bare, threat (+6 more)

### Community 77 - "stt_model_version"
Cohesion: 0.19
Nodes (6): model_label(), _package_version(), stt_model_version(), yamnet_version(), model_versions(), ModelVersionTests

### Community 78 - ".start"
Cohesion: 0.14
Nodes (3): CoordinatorAudioWorker, TextDetectorWorker, set_media_classifier()

### Community 79 - "false_alarm_filters.py"
Cohesion: 0.19
Nodes (7): decode_gate_overridden(), _env_bool(), _env_float(), _env_int(), _env_optional_float(), contains_urgent_directive(), ConfigurationTests

### Community 80 - "ScriptedClassifier"
Cohesion: 0.20
Nodes (4): shadow_candidate_alert(), shadow_candidate_case_alert(), prediction(), ScriptedClassifier

### Community 81 - "write_identity"
Cohesion: 0.21
Nodes (3): write_identity(), CredentialChoiceTests, FakeClock

### Community 82 - "EchoSense School Setup Guide"
Cohesion: 0.14
Nodes (13): Adding a New Box (QR setup), Demo Commands, EchoSense School Setup Guide, Every Time At School, If Pi Does Not Connect, Network Notes, Step 1 — Turn on hotspot, Step 2 — Plug in Pi (+5 more)

### Community 83 - "categories"
Cohesion: 0.15
Nodes (13): categories, intelligence, self_harm, label, self_harm, everyday, extra, kind (+5 more)

### Community 85 - "EdgeRuntime"
Cohesion: 0.24
Nodes (3): stop_recorder(), EdgeRuntime, RuntimeAfterTests

### Community 86 - "test_false_alarm_filters.py"
Cohesion: 0.27
Nodes (4): DecodeMetrics, summarize_segments(), WordScore, DecodeMetricsTests

### Community 87 - "SemanticClassifier"
Cohesion: 0.17
Nodes (6): SemanticClassifier, Candidate assessment, Current state, Experimental semantic and context upgrade, Implemented interfaces and safety policy, Independent evaluation gate

### Community 88 - "Persistent Offline Alert Outbox"
Cohesion: 0.17
Nodes (11): Configuration, Database and schema, Health and safe inspection, HTTP classification, Persistent Offline Alert Outbox, Purpose and architecture, Retention, backup, and test reset, Retry schedule and startup recovery (+3 more)

### Community 89 - "language_restricted_engine.py"
Cohesion: 0.23
Nodes (5): _engine_class(), transcribe(), __getattr__(), _restricted_whisper_worker(), _whisper_model()

### Community 91 - "AudioAlignmentTests"
Cohesion: 0.27
Nodes (5): AudioAlignmentTests, tone(), waveform(), _stt_result(), _tone_result()

### Community 93 - "ActivationTests"
Cohesion: 0.26
Nodes (3): ActivationTests, post(), FakeResponse

### Community 94 - "media_gate"
Cohesion: 0.25
Nodes (4): _label(), media_classes_heard(), media_gate(), MediaGateTests

### Community 95 - "Raspberry Pi 5 Wi-Fi Provisioning"
Cohesion: 0.18
Nodes (10): Architecture, Capstone checking: exact phone workflow, Dashboard integration contract, Hardware validation checklist, Installation after explicit hardware approval, Raspberry Pi 5 Wi-Fi Provisioning, Read-only hardware preflight (approval gate), Rollback plan to approve before activation (+2 more)

### Community 97 - "test_tone.py"
Cohesion: 0.29
Nodes (5): get_tone_confidence_boost(), main(), make_aggressive_audio(), make_casual_audio(), show()

### Community 98 - "validate_dashboard_test_configuration"
Cohesion: 0.33
Nodes (3): DashboardTestConfigurationError, validate_dashboard_test_configuration(), DashboardTestConfigurationTests

### Community 101 - "Repository Guidelines"
Cohesion: 0.25
Nodes (7): Build, Test, and Development Commands, Coding Style & Naming Conventions, Commit & Pull Request Guidelines, Project Structure & Module Organization, Repository Guidelines, Security & Configuration, Testing Guidelines

### Community 102 - "capture.py"
Cohesion: 0.32
Nodes (4): capture_audio(), get_audio_device(), get_pyaudio_instance(), terminate_audio_system()

### Community 103 - "CLAUDE.md"
Cohesion: 0.25
Nodes (6): Configuration caveats (`config.py`), Environment setup (first-time only), Known landmines, One codebase for Mac and Pi, Pipeline architecture, Running the system

### Community 104 - "shadow_log.py"
Cohesion: 0.32
Nodes (4): _near_miss(), log_alert(), log_near_miss(), _write()

### Community 105 - "FilterConfig"
Cohesion: 0.43
Nodes (3): FilterConfig, hallucination_reason(), HallucinationTests

### Community 106 - "atay"
Cohesion: 0.25
Nodes (8): bare, everyday, extra, kind, meaning, priority, vocative, atay

### Community 107 - "roots"
Cohesion: 0.25
Nodes (8): swear_slurs, tomboy, label, roots, everyday, kind, meaning, priority

### Community 108 - "pangit"
Cohesion: 0.25
Nodes (8): end_only, everyday, exclamative, extra, kind, meaning, priority, pangit

### Community 109 - "piste"
Cohesion: 0.25
Nodes (8): bare, everyday, extra, kind, meaning, priority, vocative, piste

### Community 112 - "Multilingual sentence-model research"
Cohesion: 0.29
Nodes (6): Candidate comparison, Downloads that would require explicit approval, Evaluation implications, Evidence standard for the target languages, Multilingual sentence-model research, Recommendation

### Community 113 - "animal"
Cohesion: 0.29
Nodes (7): bare, everyday, kind, meaning, priority, vocative, animal

### Community 114 - "baho"
Cohesion: 0.29
Nodes (7): everyday, exclamative, extra, kind, meaning, priority, baho

### Community 115 - "bogo"
Cohesion: 0.29
Nodes (7): everyday, exclamative, extra, kind, meaning, priority, bogo

### Community 116 - "buang"
Cohesion: 0.29
Nodes (7): everyday, exclamative, extra, kind, meaning, priority, buang

### Community 117 - "buktot"
Cohesion: 0.29
Nodes (7): everyday, exclamative, extra, kind, meaning, priority, buktot

### Community 118 - "bungi"
Cohesion: 0.29
Nodes (7): everyday, exclamative, extra, kind, meaning, priority, bungi

### Community 119 - "buwisit"
Cohesion: 0.29
Nodes (7): bare, extra, kind, meaning, priority, vocative, buwisit

### Community 120 - "hugaw"
Cohesion: 0.29
Nodes (7): everyday, exclamative, extra, kind, meaning, priority, hugaw

### Community 121 - "itom"
Cohesion: 0.29
Nodes (7): everyday, exclamative, extra, kind, meaning, priority, itom

### Community 122 - "kutohon"
Cohesion: 0.29
Nodes (7): everyday, exclamative, extra, kind, meaning, priority, kutohon

### Community 123 - "leche"
Cohesion: 0.29
Nodes (7): bare, everyday, kind, meaning, priority, vocative, leche

### Community 124 - "linti"
Cohesion: 0.29
Nodes (7): bare, everyday, kind, meaning, priority, vocative, linti

### Community 125 - "maot"
Cohesion: 0.29
Nodes (7): everyday, exclamative, extra, kind, meaning, priority, maot

### Community 126 - "niwang"
Cohesion: 0.29
Nodes (7): everyday, exclamative, extra, kind, meaning, priority, niwang

### Community 127 - "pandak"
Cohesion: 0.29
Nodes (7): everyday, exclamative, extra, kind, meaning, priority, pandak

### Community 128 - "tambok"
Cohesion: 0.29
Nodes (7): tambok, everyday, exclamative, extra, kind, meaning, priority

### Community 129 - "yawa"
Cohesion: 0.29
Nodes (7): yawa, bare, everyday, kind, meaning, priority, vocative

### Community 130 - "amaw"
Cohesion: 0.33
Nodes (6): everyday, exclamative, kind, meaning, priority, amaw

### Community 131 - "ampon"
Cohesion: 0.33
Nodes (6): everyday, extra, kind, meaning, priority, ampon

### Community 132 - "bati"
Cohesion: 0.33
Nodes (6): everyday, extra, kind, meaning, priority, bati

### Community 133 - "bayot"
Cohesion: 0.33
Nodes (6): everyday, extra, kind, meaning, priority, bayot

### Community 134 - "bisaya"
Cohesion: 0.33
Nodes (6): everyday, extra, kind, meaning, priority, bisaya

### Community 135 - "bobo"
Cohesion: 0.33
Nodes (6): everyday, exclamative, kind, meaning, priority, bobo

### Community 136 - "dakog ilong"
Cohesion: 0.33
Nodes (6): everyday, extra, kind, meaning, priority, dakog ilong

### Community 137 - "gaga"
Cohesion: 0.33
Nodes (6): bare, kind, meaning, priority, vocative, gaga

### Community 138 - "gago"
Cohesion: 0.33
Nodes (6): bare, kind, meaning, priority, vocative, gago

### Community 139 - "punyeta"
Cohesion: 0.33
Nodes (6): bare, kind, meaning, priority, vocative, punyeta

### Community 142 - "Raspberry Pi Device Authentication"
Cohesion: 0.40
Nodes (4): Delivery behavior, Provision the one-time key, Raspberry Pi Device Authentication, Rotation, disablement, and rollback

### Community 143 - "EchoSense Edge Severity Rules"
Cohesion: 0.40
Nodes (4): Compatibility, EchoSense Edge Severity Rules, Explainability, Known limitations

### Community 144 - "pulubi"
Cohesion: 0.40
Nodes (5): everyday, kind, meaning, priority, pulubi

### Community 145 - "sumbagon"
Cohesion: 0.40
Nodes (5): sumbagon, extra, kind, meaning, priority

### Community 146 - "outbox_admin.py"
Cohesion: 0.60
Nodes (3): main(), _outbox(), _safe_record()

### Community 147 - "Dashboard Test Mode"
Cohesion: 0.50
Nodes (3): Dashboard Test Mode, Local end-to-end command, Safety boundary

### Community 148 - "False-alert validation"
Cohesion: 0.50
Nodes (3): Controlled EMEET microphone session, False-alert validation, Raspberry Pi 5 checklist

### Community 149 - "bunalan"
Cohesion: 0.50
Nodes (4): kind, meaning, priority, bunalan

### Community 150 - "latigohon"
Cohesion: 0.50
Nodes (4): kind, meaning, priority, latigohon

### Community 151 - "lumpagon"
Cohesion: 0.50
Nodes (4): kind, meaning, priority, lumpagon

### Community 152 - "sagpaon"
Cohesion: 0.50
Nodes (4): sagpaon, kind, meaning, priority

### Community 153 - "sipaon"
Cohesion: 0.50
Nodes (4): sipaon, kind, meaning, priority

### Community 154 - "tadyakan"
Cohesion: 0.50
Nodes (4): tadyakan, kind, meaning, priority

### Community 156 - "roots"
Cohesion: 0.67
Nodes (3): label, roots, body_shaming

## Knowledge Gaps
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `check_transcript()` connect `check_transcript` to `evaluate_testset.py`, `severity.py`, `school_config.py`, `text_context.py`, `MultilingualPhraseDetectionTests`, `blacklist.py`, `CLAUDE.md`, `transcript_quality.py`, `test_false_positive_provenance.py`, `TextDetector`, `fuse_event_evidence`, `replace_backend_terms`, `_run_transcription_cycle`, `whisper_stt.py`, `test_severity_and_laughter.py`, `test_school_config.py`, `sent`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `TextDetector` (e.g. with `DetectedTermProvenance` and `MatchedTermEvidence`) actually correct?**
  _`TextDetector` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Should `evaluate_testset.py` be split into smaller, more focused modules?**
  _Cohesion score 0.053410893707033315 - nodes in this community are weakly interconnected._
- **Why does `TextDetector` connect `TextDetector` to `evaluate_testset.py`, `http_client.py`, `numpy`, `EventCoordinator`, `fuse_event_evidence`, `.from_float32_samples`, `CoordinatorHarness`, `test_school_config.py`, `pathlib`, `test_semantic_upgrade.py`, `.run_utterance`, `test_false_positive_provenance.py`, `runtime.py`, `evaluate_semantic_upgrade.py`, `RealtimeSTTAudioEventAdapter`, `SemanticAnalyzer`, `test_accuracy_context.py`, `.start`, `ScriptedClassifier`, `EdgeRuntime`, `test_false_alarm_filters.py`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `AlertOutbox` (e.g. with `AlertDeliveryService` and `OutboxDeliveryWorker`) actually correct?**
  _`AlertOutbox` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Should `severity.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06078316773816481 - nodes in this community are weakly interconnected._
- **Why does `ProvisioningController` connect `test_wifi_setup.py` to `PortalRouteTests`, `WifiSetupService`, `credentials.py`, `ControllerRecoveryTests`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._