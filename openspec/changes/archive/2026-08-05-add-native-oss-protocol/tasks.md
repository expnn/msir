## 1. Dependencies

- [x] 1.1 Add `services-oss` feature to opendal in `Cargo.toml` (opendal 0.56: `services-oss = ["dep:opendal-service-oss"]`)
- [x] 1.2 Verify `opendal::services::Oss` builder is exported and its field methods (`bucket`, `endpoint`, `root`, `region`, `access_key_id`, `secret_access_key`) match the current S3 builder usage

## 2. URL parsing and scheme routing

- [x] 2.1 Rewrite scheme dispatch in `build_store` (`src/lib.rs:214`): `oss://` → Oss branch, `s3://` → S3 branch, `http(s)://` → S3-with-URL-endpoint branch, `file://` → FilesystemStore, anything else → unsupported-scheme error
- [x] 2.2 Implement authority-as-bucket parsing for `oss://` and `s3://`: bucket from `parsed.host_str()`, root from `parsed.path()`; empty authority returns "missing bucket" parse error (kills triple-slash `oss:///bucket/path`)
- [x] 2.3 Keep existing `http(s)://` path-first-segment bucket parsing unchanged (`src/lib.rs:248-257`)
- [x] 2.4 Remove the old endpoint-in-host interpretation for `oss://` (no `oss://host/bucket/path` compatibility, no fallback to `DEFAULT_OSS_ENDPOINT` for `oss://`)
- [x] 2.5 Add unit tests for URL parsing: valid `oss://bucket/root`, valid `s3://bucket/root`, triple-slash rejection, old-style `oss://host/bucket/path` parsed as bucket `host`, userinfo extraction

## 3. Native OSS backend branch

- [x] 3.1 Implement the `oss://` branch building `Oss::default().bucket(..).endpoint(..).root(..)` with `OSS_ENDPOINT` required (error if unset), `OSS_REGION` defaulting to `cn-hangzhou`
- [x] 3.2 Wire credentials: `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` env vars, URL userinfo precedence
- [x] 3.3 Reuse the existing credential preflight (`op.list_with("/")`, `src/lib.rs:304-332`) for the OSS branch, mapping permission/not-found/connection errors to actionable messages
- [x] 3.4 Confirm the native OSS backend uses virtual-host style automatically (verify via request capture or debug logs); no `enable_virtual_host_style` needed for `Oss`
- [x] 3.5 Confirm `OssStoreWrapper` still applies to the native backend (root-as-common-prefix quirk); if the quirk is absent, record finding and leave wrapper in place (removal is a separate decision)

## 4. S3 branch environment namespace

- [x] 4.1 In the `s3://` branch, read `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` / `AWS_REGION` (default `auto`) and `AWS_ENDPOINT_URL` (optional); no fallback to `OSS_*` vars
- [x] 4.2 Confirm `http(s)://` branch uses the same `AWS_*` namespace and URL-host endpoint (behavior unchanged from today except env var names)
- [x] 4.3 Remove the old `OSS_*` env fallback in the S3/http(s) branches (`src/lib.rs:259-292`) and update error messages that referenced `OSS_ACCESS_KEY_ID` for S3 scenarios

## 5. Failure diagnostics

- [x] 5.1 Implement the diagnostic probe: on preflight/store failure, send one lightweight HEAD (or reuse the failed preflight response) and read `Server: AliyunOSS` / `x-oss-request-id` headers
- [x] 5.2 Build protocol-mismatch hints: AliyunOSS found but user used `s3://`/`http(s)://` → suggest `oss://bucket/path` + `OSS_ENDPOINT`; non-Aliyun server found but user used `oss://` → suggest `s3://` or `http(s)://`; inconclusive → generic error
- [x] 5.3 Ensure the probe never retries, never switches backends, and only augments the error message (satisfies one-shot diagnostic requirement)
- [x] 5.4 Add unit tests for the three diagnostic branches (Aliyun detected, non-Aliyun detected, inconclusive) using a mock HTTP endpoint

## 6. Verification and performance experiment

- [x] 6.1 Build and run existing test suite; confirm no regressions in `file://` and `http(s)://` paths
- [ ] 6.2 Run experiment 1 (connection-pool cold start): same reader reads N samples vs new reader per sample; record throughput to quantify per-operator pool warmup cost
- [ ] 6.3 Run experiment 2 (path-style vs virtual-host) on a real `*.aliyuncs.com` endpoint with `services-s3`, toggling `enable_virtual_host_style()`; verify whether Aliyun rejects path-style and record latency/throughput delta
- [ ] 6.4 Run experiment 3 (services-s3 vs services-oss) on the same real Aliyun endpoint, both virtual-host; record throughput/P50/P99 to test the "S3 protocol is the bottleneck" hypothesis
- [ ] 6.5 Run experiment 4 (HTTP/2 vs HTTP/1.1) with `http1_only()` if experiment 3 shows no clear protocol gap
- [ ] 6.6 Record experiment results in the change (or a doc) and update design.md's conclusion on whether native OSS protocol is worth it; if the "empty data" assertion from `src/lib.rs:283-285` is disproven, note it (potential upstream OpenDAL issue)

## 7. Docs and migration

- [x] 7.1 Update `build_store` doc comments to document the new scheme table and breaking change
- [x] 7.2 Update any Python-facing error messages/docstrings that reference the old `oss://host/bucket/path` form or `OSS_*` env vars for S3 scenarios
- [x] 7.3 Record the migration guidance (S3-compatible stores → `s3://`/`https://`; real Aliyun OSS → `oss://` + `OSS_ENDPOINT`) in the change artifacts for release notes
