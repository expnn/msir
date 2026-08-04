## ADDED Requirements

### Requirement: URL scheme determines protocol backend

The store builder SHALL map the URL scheme to a specific storage protocol backend: `oss://` uses the Aliyun OSS native protocol, `s3://` and `http(s)://` use the S3-compatible protocol, and `file://` uses the local filesystem.

#### Scenario: oss scheme selects native OSS backend

- **WHEN** `build_store` receives a URL with scheme `oss://`
- **THEN** it constructs an OpenDAL `Oss` operator (native Aliyun OSS protocol)

#### Scenario: s3 scheme selects S3 backend

- **WHEN** `build_store` receives a URL with scheme `s3://`
- **THEN** it constructs an OpenDAL `S3` operator (S3-compatible protocol)

#### Scenario: http and https schemes select S3 backend

- **WHEN** `build_store` receives a URL with scheme `http://` or `https://`
- **THEN** it constructs an OpenDAL `S3` operator with the URL host as the explicit endpoint

#### Scenario: file scheme selects filesystem store

- **WHEN** `build_store` receives a URL with scheme `file://`
- **THEN** it constructs a `FilesystemStore` rooted at the URL path

### Requirement: Bucket is required in the URL authority

For `oss://` and `s3://` URLs, the bucket name SHALL be taken from the URL authority (host field) and MUST be non-empty. An empty authority SHALL be rejected at parse time.

#### Scenario: bucket in authority

- **WHEN** `build_store` receives `oss://my-bucket/data/zarr/root` or `s3://my-bucket/data/zarr/root`
- **THEN** it resolves bucket as `my-bucket` and root as `/data/zarr/root`

#### Scenario: empty authority is rejected

- **WHEN** `build_store` receives `oss:///bucket/path` or `s3:///bucket/path` (triple-slash, empty host)
- **THEN** it returns a parse error indicating the bucket is missing and does not attempt any connection

#### Scenario: endpoint in authority is no longer supported

- **WHEN** `build_store` receives `oss://endpoint.example.com/bucket/path` (old-style endpoint-in-host URL)
- **THEN** it treats `endpoint.example.com` as the bucket name (per the authority rule), and the resulting connection error is augmented by the protocol-mismatch diagnostic

### Requirement: Endpoint comes from environment variables

For `oss://` URLs the endpoint SHALL come from the `OSS_ENDPOINT` environment variable and is REQUIRED. For `s3://` URLs the endpoint SHALL come from the `AWS_ENDPOINT_URL` environment variable and is optional (defaulting to AWS standard endpoints). The endpoint SHALL NOT be embedded in `oss://` or `s3://` URLs.

#### Scenario: oss endpoint from environment

- **WHEN** `build_store` receives `oss://my-bucket/root` and `OSS_ENDPOINT` is set
- **THEN** it uses the `OSS_ENDPOINT` value as the endpoint

#### Scenario: oss endpoint missing

- **WHEN** `build_store` receives `oss://my-bucket/root` and `OSS_ENDPOINT` is unset
- **THEN** it returns an error stating that `OSS_ENDPOINT` is required for `oss://` URLs

#### Scenario: s3 endpoint from environment

- **WHEN** `build_store` receives `s3://my-bucket/root` and `AWS_ENDPOINT_URL` is set
- **THEN** it uses the `AWS_ENDPOINT_URL` value as the endpoint

### Requirement: Credential environment variable namespaces

The `oss://` scheme SHALL read credentials from `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` / `OSS_REGION`. The `s3://` and `http(s)://` schemes SHALL read credentials from `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` / `AWS_REGION`. URL userinfo (`oss://key:secret@bucket/root`) SHALL take precedence over environment variables.

#### Scenario: oss credentials from environment

- **WHEN** `build_store` receives `oss://my-bucket/root` and `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` are set
- **THEN** it uses those values as credentials

#### Scenario: s3 credentials from environment

- **WHEN** `build_store` receives `s3://my-bucket/root` and `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are set
- **THEN** it uses those values as credentials

#### Scenario: userinfo takes precedence

- **WHEN** `build_store` receives `oss://user:pass@my-bucket/root`
- **THEN** it uses `user` and `pass` as credentials regardless of environment variables

### Requirement: No cross-namespace credential fallback

The `s3://` and `http(s)://` schemes SHALL NOT fall back to `OSS_*` environment variables when `AWS_*` variables are unset, and the `oss://` scheme SHALL NOT fall back to `AWS_*` variables.

#### Scenario: s3 without AWS vars does not read OSS vars

- **WHEN** `build_store` receives `s3://my-bucket/root` with only `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` set
- **THEN** it constructs the operator without those credentials (no silent fallback)

#### Scenario: oss without OSS vars does not read AWS vars

- **WHEN** `build_store` receives `oss://my-bucket/root` with only `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` set
- **THEN** it constructs the operator without those credentials (no silent fallback)

### Requirement: Old oss endpoint-in-host URLs are a hard break

URLs of the form `oss://host/bucket/path` SHALL NOT be interpreted as endpoint-in-host; there SHALL be no compatibility alias, no deprecated fallback, and no warning-based dual interpretation. The only assistance is the failure diagnostic.

#### Scenario: no compatibility mode

- **WHEN** `build_store` receives `oss://oss-cn-hangzhou.aliyuncs.com/my-bucket/root`
- **THEN** it parses `oss-cn-hangzhou.aliyuncs.com` as the bucket name and makes no attempt to interpret it as an endpoint
