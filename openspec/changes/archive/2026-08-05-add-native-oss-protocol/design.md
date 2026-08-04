## Context

`msir` 是 Rust + PyO3 扩展，通过 `zarrs_opendal::AsyncOpendalStore` 读取 zarr 数据。当前 `build_store`（`src/lib.rs:214`）将 `oss://`、`http://`、`https://` 三个 scheme 统一映射到 OpenDAL `services-s3`（S3 兼容协议），endpoint 放在 URL host 位置，bucket 取 path 首段。

三个已确认的现状问题：
1. **协议选择一刀切**：真阿里 OSS 用户也被迫走 S3 仿真协议，用户实测比预期慢。OpenDAL 有原生 `services-oss` 后端（阿里 OSS V1/V2/V4 签名、`x-oss-*` header 命名空间、CRC-64 等原生特性）但从未使用。
2. **URL 规范混乱**：`oss://[host]/bucket/path` 的 host 可省略，产生 `oss:///bucket/path` 畸形 URL。生态惯例（AWS CLI、ossutil、fsspec、rclone）都是 `scheme://bucket/key`（bucket 在 authority），endpoint 走外部配置。
3. **代码注释断言存疑**：`src/lib.rs:283-285` 声称"services-oss 对 S3 兼容 endpoint 请求会成功建立连接但返回空数据"。OpenDAL issue/文档中无此记录，理论预期是 403 签名不匹配而非空数据。该断言是当前"统一用 services-s3"的决策依据，需实验验证。

关键事实（源码验证）：
- OpenDAL 0.56 S3 后端默认 **path-style**（`backend.rs:430` doc 注释），需显式 `enable_virtual_host_style()` 才切 virtual-host。阿里 OSS **只支持 virtual-host**，path-style 请求返回 `SecondLevelDomainForbidden`。当前 msir 未调用该 API。
- OpenDAL 0.56 `services-oss` feature 存在（`services-oss = ["dep:opendal-service-oss"]`），签名用 `reqsign-aliyun-oss`。
- 连接池挂在 reqwest `Client` 上，`Operator` 持有 `Client`；msir 单个 `AstroImageReader` 内复用 store/Operator（`src/lib.rs:419-421` 注释确认），但每次构造 reader 都会新建 Operator → 空连接池。

## Goals / Non-Goals

**Goals:**
- `oss://` 显式走阿里 OSS 原生协议（`services-oss`），`s3://` 显式走 S3 协议（`services-s3`），`http(s)://` 走 S3 协议 + URL 内嵌 endpoint，`file://` 走本地。
- 规范 URL 形式为 `scheme://bucket/key`（bucket 在 authority，endpoint 走环境变量），消灭三斜杠畸形 URL。
- 规范鉴权命名空间：`oss://` 读 `OSS_*`，`s3://`/`http(s)://` 读 `AWS_*`，URL userinfo 优先。
- 访问失败时探测并输出协议不匹配诊断信息（不自动切换）。
- 用实验数据验证"S3 协议是否真的比原生 OSS 慢"，以及 path-style/virtual-host、连接池、HTTP 版本等混淆因子。

**Non-Goals:**
- 不自动探测存储类型并在运行时切换协议（用户已明确否决，改为显式 schema）。
- 不为旧 `oss://endpoint/bucket/path` URL 做兼容（选项 C：硬切）。
- 不做 `s3://` 对 `OSS_*` 环境变量的 fallback（硬切）。
- 不实现写入路径——当前 msir 是只读 reader，仅需读。
- 不重写连接池/HTTP 层优化（列为实验变量，不在本 change 强制实现，除非实验证实其必要）。

## Decisions

### D1: scheme → 后端映射

| scheme | 后端 | bucket 来源 | endpoint 来源 | 凭证来源 |
|---|---|---|---|---|
| `oss://` | `services-oss` | authority（必填） | `OSS_ENDPOINT` env（必填） | `OSS_ACCESS_KEY_ID`/`OSS_ACCESS_KEY_SECRET` env 或 URL userinfo |
| `s3://` | `services-s3` | authority（必填） | `AWS_ENDPOINT_URL` env（可选，缺省 AWS 标准 endpoint） | `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN` env 或 URL userinfo |
| `http(s)://` | `services-s3` | path 首段 | URL host | 同 `s3://` |
| `file://` | `FilesystemStore` | — | — | — |

**替代方案**：运行时探测（`Server: AliyunOSS` header）自动选后端。否决理由：探测请求本身有成本、非阿里存储可能被误判、隐式行为难排查。生态先例（fsspec、rclone、OpenDAL）都是显式 scheme 映射。失败时的诊断探测（D4）保留探测思想但只用于报错。

### D2: URL 解析规则

- `oss://bucket/key` 与 `s3://bucket/key`：bucket 取自 `parsed.host_str()`（必填，空则报 "missing bucket"），root 取自 `parsed.path()`。`url` crate 对非特殊 scheme 会解析 authority，`oss://bucket/key` 中 `bucket` 恰好落在 host 字段——正好符合生态惯例。
- `http(s)://host[:port]/bucket/key`：endpoint = host[:port]，bucket = path 首段，root = 其余 path（保持现有逻辑，`src/lib.rs:248-257`）。
- userinfo `oss://key:secret@bucket/key`：优先于环境变量。
- **BREAKING**：`oss://endpoint/bucket/key` 不再解析为 endpoint 在 host；`endpoint` 会被当作 bucket 名，后续连接报错时由诊断信息（D4）指引用户改 `https://endpoint/bucket/key`。
- 三斜杠 `oss:///bucket/path`：host 为空 → 解析期直接报 "missing bucket"。

### D3: 鉴权与 endpoint 环境变量命名空间

沿用阿里官方（ossutil/oss2）与 AWS 标准命名，避免发明：
- `oss://` → `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` / `OSS_ENDPOINT` / `OSS_REGION`（默认 `cn-hangzhou`）
- `s3://` / `http(s)://` → `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` / `AWS_ENDPOINT_URL` / `AWS_REGION`（默认 `auto`）
- 不保留旧代码 `oss://` 分支读 `OSS_*` 当 S3 用的行为。S3 场景用户必须改用 `s3://`/`http(s)://` + `AWS_*`。

### D4: 失败诊断探测

存储连接/凭证校验失败（现有 `build_store` 已有凭证预检 `op.list_with("/")`，`src/lib.rs:304-332`）时增强为：
1. 向 endpoint 发一次 HEAD 请求（复用已有预检的 list 调用，或在其失败后追加）。
2. 检查响应头 `Server: AliyunOSS` 与 `x-oss-request-id`。
3. 根据探测结果拼装诊断错误：
   - `Server: AliyunOSS` 且用户用了 `s3://`/`http(s)://` → "endpoint 是阿里 OSS，原生协议请用 `oss://bucket/key` + `OSS_ENDPOINT`"。
   - 非 AliyunOSS 且用户用了 `oss://` → "endpoint 不是阿里 OSS，`oss://` 仅支持真阿里 OSS；S3 兼容存储请用 `s3://` 或 `http(s)://`"。
   - 无法判定 → 通用连接错误。
- 纯诊断，不自动切换协议，不改变原有权限/not-found 错误分类。

### D5: 原生 OSS 后端接入方式

- `Cargo.toml` opendal 增加 feature `services-oss`。
- `build_store` 中 `oss://` 分支构建 `Oss::default().bucket(...).endpoint(...).root(...)`，等价于现有 S3 builder 的字段设置。
- `services-oss` 后端天然 virtual-host 寻址，无需额外配置；这同时规避了阿里 OSS 拒 path-style 的问题。
- `OssStoreWrapper`（`src/lib.rs:116`，处理 ListObjectsV2 返回根 common_prefix 的 quirk）保留，但用实验确认原生后端是否还有此 quirk；若无，后续可移除（记入 tasks 作为独立验证项）。

### D6: 实验验证协议差异（Q1 的证明路径）

设计对照实验隔离四个混淆因子，判定"S3 协议慢"的真实来源：
1. **连接池冷启动**：同一 reader 连续读 N 样本 vs 每样本新建 reader。
2. **path-style vs virtual-host**：同 `services-s3`，同 endpoint，切换 `enable_virtual_host_style()`。
3. **services-s3 vs services-oss**：都 virtual-host，真阿里 endpoint，这是核心对照。
4. **HTTP/2 vs HTTP/1.1**：`http1_only()`（OpenDAL 官方文档建议大文件场景）。

必须用真 `*.aliyuncs.com` endpoint。ZhejiangLab 内部存储不是阿里 OSS，无法测原生协议。

## Risks / Trade-offs

- **[BREAKING 迁移阵痛] 现有 `oss://endpoint/bucket/path` 用户全部失效** → 硬切是用户明确决策（选项 C）；靠 D4 诊断信息在报错时指路，降低困惑；在 release notes / Python 侧 docstring 写明迁移说明。
- **["空数据"断言可能反转] 若实测证明 services-oss 对 S3 兼容 endpoint 确实静默返回空（而非 403）** → 这是 OpenDAL 上游 bug，应报 issue；诊断探测（D4）可兜底指引用户换 scheme。当前先按 403 预期设计，实验留验证项。
- **[OSS_REGION 默认值] `cn-hangzhou` 作为默认 region 对非杭州用户错误** → 仅影响签名 scope 不参与实际路由；要求 `OSS_ENDPOINT` 必填已隐含 region 信息；文档明确可覆盖。
- **[原生后端缺 S3 兼容场景的灵活性] `services-oss` 只认 virtual-host，无法连 S3 兼容存储** → 这正是显式 scheme 的意义：`oss://` 就是给真阿里 OSS 的，S3 兼容存储用 `s3://`/`http(s)://`。
- **[连接池冷启动] 每次构造 reader 新建 Operator → 空连接池** → 本 change 不做全局池缓存（见 Non-Goals），但实验 1 会量化其影响；若证实显著，记为后续独立 change。

## Migration Plan

1. 在 Python 侧（外部包）与 release notes 中声明 `oss://` 语义变更。
2. 现有 `oss://endpoint/bucket/path` 用户迁移指引：
   - S3 兼容存储（如 ZhejiangLab）→ `https://endpoint/bucket/path` 或 `s3://bucket/path` + `AWS_ENDPOINT_URL`。
   - 真阿里 OSS → `oss://bucket/path` + `OSS_ENDPOINT` env。
3. 环境变量迁移：`OSS_ACCESS_KEY_ID`/`OSS_ACCESS_KEY_SECRET` 仅剩 `oss://` 场景使用；S3 场景设置 `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`。
4. 回滚：仅 Rust 扩展，回滚到旧版本即可恢复旧 URL 语义（无数据迁移风险，纯协议层变更）。

## Open Questions

- 真阿里 OSS bucket 是否可提供用于实验？实验需要 `*.aliyuncs.com` endpoint + 凭证 + 测试数据。
- `OssStoreWrapper` 的 root-prefix quirk 在原生后端是否还存在？（实验 3 顺带验证）
- `AWS_ENDPOINT_URL` 与 `OSS_ENDPOINT` 是否允许 URL 内嵌 endpoint 作为例外？（当前设计：不允许，一律走 env；保持生态一致）
