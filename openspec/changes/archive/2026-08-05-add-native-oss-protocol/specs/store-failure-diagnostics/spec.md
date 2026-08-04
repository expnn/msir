## ADDED Requirements

### Requirement: Diagnose protocol mismatch on store failure

When store construction or its preflight check fails, the system SHALL perform one lightweight probe against the endpoint and, when the probe identifies the storage server type, augment the error with a protocol-mismatch hint. The system SHALL NOT automatically switch protocols based on the probe.

#### Scenario: Aliyun OSS reached via s3/http URL

- **WHEN** a `s3://` or `http(s)://` store fails and the endpoint probe detects `Server: AliyunOSS` (or `x-oss-request-id` present)
- **THEN** the error message includes a hint that the endpoint is Aliyun OSS and suggests using `oss://bucket/path` with `OSS_ENDPOINT` for the native protocol

#### Scenario: non-Aliyun S3 store reached via oss URL

- **WHEN** an `oss://` store fails and the endpoint probe detects a non-Aliyun server (no `Server: AliyunOSS`, no `x-oss-request-id`)
- **THEN** the error message includes a hint that the endpoint does not appear to be Aliyun OSS and suggests using `s3://` or `http(s)://` for S3-compatible storage

#### Scenario: probe inconclusive

- **WHEN** a store fails and the endpoint probe times out or returns no identifiable server signature
- **THEN** the error message falls back to the generic connection/config error without a protocol hint

### Requirement: Probe is one-shot and diagnostic only

The diagnostic probe SHALL be a single lightweight request (HEAD or the already-performed preflight list call), SHALL NOT be retried, SHALL NOT change the store backend, and SHALL only add a hint to the error message.

#### Scenario: no backend switch on detection

- **WHEN** a probe detects a type mismatch (e.g., Aliyun OSS behind an `s3://` URL)
- **THEN** the store still fails with the original error; only the message text is augmented

#### Scenario: probe does not retry

- **WHEN** the probe request fails
- **THEN** no further probe or retry is performed and the original error is returned
