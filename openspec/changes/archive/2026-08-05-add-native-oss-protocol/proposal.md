## Why

当前 `msir` 将所有对象存储（`oss://`、`http://`、`https://`）统一映射到 OpenDAL 的 `services-s3`（S3 兼容协议），即使目标是真正的阿里云 OSS。用户实测 S3 协议读取**比预期慢**，怀疑协议层有开销。OpenDAL 提供原生 `services-oss` 后端（阿里 OSS 原生协议），但当前代码从未使用——而且代码注释中"services-oss 对 S3 兼容 endpoint 返回空数据"的断言缺乏证据支撑，值得重新验证。同时，现有 URL 规范允许 `oss:///bucket/path`（三斜杠）这类畸形 URL，host 与 bucket 的语义混淆是混乱的根源。

## What Changes

- **BREAKING**: `oss://` scheme 语义从"S3 协议 + endpoint 在 host"改为"**阿里 OSS 原生协议 + bucket 在 authority**"。`oss://endpoint/bucket/path`（endpoint 在 host）不再受支持，改用 `https://endpoint/bucket/path`（S3 协议）或 `oss://bucket/path` + `OSS_ENDPOINT` 环境变量（原生 OSS）。硬切换，不做兼容，不保留旧 env fallback。
- 新增 `s3://` scheme，显式声明 S3 协议：`s3://bucket/key`，endpoint 从 `AWS_ENDPOINT_URL` 读取。
- `http://` / `https://` scheme 保持现有语义（S3 协议 + endpoint 在 host + bucket 在 path 首段），补齐 `http://` 的正式文档化支持。
- 新增鉴权与 endpoint 命名空间规则：`oss://` 读 `OSS_*` 环境变量，`s3://`/`http(s)://` 读 `AWS_*` 环境变量，URL userinfo 覆盖优先。
- 存储访问失败时，进行一次探测（HEAD endpoint，检查 `Server: AliyunOSS` 响应头等信号），打印可能的协议不匹配诊断信息——只诊断，不自动切换协议。
- **BREAKING**: 修复 URL 解析：bucket 作为 authority 为必填项，消灭 `oss:///bucket/path` 三斜杠畸形 URL（解析期直接报 "missing bucket"）。
- 增加 OpenDAL `services-oss` feature 依赖。
- 通过实验数据验证"原生 OSS 协议是否真的降低读取开销"（S3 协议 vs 原生协议、path-style vs virtual-host、连接池复用、HTTP/2 vs HTTP/1.1 四个维度）。

## Capabilities

### New Capabilities

- `store-url-schemes`: URL scheme（`oss://`/`s3://`/`http://`/`https://`/`file://`）到协议后端与配置来源的映射规则，含 bucket/endpoint/凭证解析。
- `oss-native-read`: 使用 OpenDAL `services-oss` 通过阿里 OSS 原生协议读取数据的能力。
- `store-failure-diagnostics`: 存储访问失败时的探测与协议不匹配诊断信息输出。

### Modified Capabilities

<!-- 无既有 spec，全部为新增 capability -->

## Impact

- `src/lib.rs`: `build_store` 的 URL 解析与后端选择逻辑重写；新增 `s3://` 分支、`oss://` 原生协议分支；`DEFAULT_OSS_ENDPOINT` 常量的处理方式改变。
- `Cargo.toml`: opendal 增加 `services-oss` feature。
- 依赖: `opendal-service-oss`、`reqsign-aliyun-oss`（传递依赖）。
- Python 端调用方：`oss://endpoint/bucket/path` 形式的 URL 全部失效，需要迁移到新 scheme；`OSS_ACCESS_KEY_ID`/`OSS_ACCESS_KEY_SECRET` 环境变量仅对 `oss://` 生效，S3 场景需改用 `AWS_*`。
- 实验验证：需要真 `*.aliyuncs.com` endpoint + bucket 进行对比测量。
