## ADDED Requirements

### Requirement: Read zarr data via native OSS protocol

The `oss://` scheme SHALL read data using the OpenDAL `services-oss` backend (Aliyun OSS native protocol with OSS V1/V2/V4 signing and `x-oss-*` request headers), enabling full read support for zarr datasets stored in Aliyun OSS.

#### Scenario: zarr reads via native OSS

- **WHEN** `AstroImageReader` is constructed with an `oss://my-bucket/path/to/zarr/root` URL and valid `OSS_ENDPOINT` / `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`
- **THEN** all zarr metadata and chunk reads succeed and return the same data as the S3-compatible protocol would

#### Scenario: bucket and root resolution

- **WHEN** `build_store` receives `oss://my-bucket/data/zarr/root`
- **THEN** the OSS operator is configured with bucket `my-bucket`, root `/data/zarr/root`, and the endpoint from `OSS_ENDPOINT`

### Requirement: Virtual-host style addressing for OSS

The `oss://` backend SHALL address objects in virtual-host style (bucket as subdomain of endpoint), which is the only style Aliyun OSS supports.

#### Scenario: native OSS uses virtual-host

- **WHEN** an `oss://` operator issues a request
- **THEN** the request targets `my-bucket.<endpoint-host>/<key>` (bucket in the Host header / virtual-host style), never `<endpoint-host>/my-bucket/<key>`

### Requirement: Credential preflight check for OSS

The `oss://` branch SHALL perform a credential and connectivity preflight check before returning the store, so that wrong credentials, wrong bucket, or unreachable endpoints fail loudly with actionable errors instead of surfacing as confusing downstream zarr errors.

#### Scenario: access denied is reported clearly

- **WHEN** the preflight list call fails with a permission error
- **THEN** `build_store` returns an error naming `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` (or the URL userinfo) as the suspected cause

#### Scenario: bucket not found is reported clearly

- **WHEN** the preflight list call fails with a not-found error
- **THEN** `build_store` returns an error pointing at the bucket name and path in the URL

#### Scenario: unreachable endpoint is reported clearly

- **WHEN** the preflight list call fails with a network/connection error
- **THEN** `build_store` returns an error pointing at `OSS_ENDPOINT` and network connectivity

### Requirement: Native OSS list quirk handling retained

The existing list-result normalization (`OssStoreWrapper`) SHALL continue to apply to the native OSS backend so that datasets whose root appears as a common prefix are listed correctly.

#### Scenario: root-as-common-prefix is normalized

- **WHEN** a listing via the `oss://` backend returns the root itself as a common prefix entry (an Aliyun OSS quirk)
- **THEN** the wrapper normalizes the result so the zarr reader does not misinterpret the listing
