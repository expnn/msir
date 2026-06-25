use ndarray::{Array3, Array4, Axis};
use numpy::{IntoPyArray, PyArray4, PyReadonlyArray1};
use opendal::{Operator, services::S3};
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rust_lapper::{Interval, Lapper};
use std::env;
use std::future::Future;
use std::sync::{Arc, OnceLock};
use thiserror::Error;
use tokio::runtime::Runtime;
use url::{ParseError, Url};
use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;
use zarrs::storage::storage_adapter::async_to_sync::{
    AsyncToSyncBlockOn, AsyncToSyncStorageAdapter,
};
use zarrs::storage::{
    byte_range::ByteRangeIterator, ListableStorageTraits, MaybeBytes, MaybeBytesIterator,
    ReadableListableStorageTraits, ReadableStorageTraits, StorageError, StoreKey, StoreKeys,
    StoreKeysPrefixes, StorePrefix,
};
use zarrs_opendal::AsyncOpendalStore;

// noinspection HttpUrlsUsage
const DEFAULT_OSS_ENDPOINT: &str = "http://oss-cn-hangzhou-zjy-d01-a.res.cloud.zhejianglab.com";

type Example<'py> = PyResult<(
    Bound<'py, PyArray4<f32>>,
    Option<Bound<'py, PyArray4<bool>>>,
    Option<Bound<'py, PyArray4<f32>>>,
)>;

type ExampleArray3 = PyResult<(Array3<f32>, Option<Array3<bool>>, Option<Array3<f32>>)>;
type BlockResult = PyResult<(Array4<f32>, Option<Array4<bool>>, Option<Array4<f32>>)>;
type IndexType = PyResult<(
    ReadableStore,
    Lapper<usize, IntervalData>,
    Vec<String>,
    usize,
    usize,
)>;

#[derive(Error, Debug)]
pub enum MsirError {
    #[error("Zarr error: {0}")]
    Zarr(String),
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(usize),
    #[error("Invalid data shape: {0}")]
    InvalidShape(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<MsirError> for PyErr {
    fn from(err: MsirError) -> PyErr {
        match err {
            MsirError::Zarr(e) => PyIOError::new_err(e.to_string()),
            MsirError::IndexOutOfBounds(e) => PyIndexError::new_err(e.to_string()),
            MsirError::InvalidShape(e) => PyValueError::new_err(e.to_string()),
            MsirError::Io(e) => PyIOError::new_err(e.to_string()),
        }
    }
}

/// 共享的 tokio runtime，供 AsyncToSyncStorageAdapter 使用。
///
/// 该 runtime 在模块初始化阶段（参见下方 `#[pymodule_init]` 标记的函数）通过
/// `OnceLock::set` 一次性写入。之后所有访问都只调用 `OnceLock::get`，不会进入
/// 任何初始化闭包，因此不存在 PyO3 FAQ 中提到的 `get_or_init` 在 GIL 下执行
/// 阻塞闭包所引发的死锁风险。
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// 获取已初始化的 tokio runtime 引用。
///
/// 模块初始化（`init_module`）成功意味着 `RUNTIME` 已经被 `set`，Python 端只要
/// `import msir` 成功，本函数即永远返回有效引用。初始化失败会在模块导入阶段
/// 直接抛出 `ImportError`，不会进入此函数。
fn runtime() -> &'static Runtime {
    // 仅调用 `get`，不使用 `get_or_init`，因此与 PyO3 GIL 之间不存在
    // OnceLock 初始化闭包可能引入的死锁问题。
    RUNTIME
        .get()
        .expect("tokio runtime must be initialized during msir module import")
}

/// 将 tokio runtime 适配到 zarrs 的 AsyncToSyncBlockOn trait。
struct TokioBlockOn;

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: Future>(&self, future: F) -> F::Output {
        let rt = runtime();
        let _guard = rt.enter();
        rt.block_on(future)
    }
}

/// zarrs 同步可读 + 可列举的统一 store 类型别名。
/// `child_groups` 需要 `ListableStorageTraits`，所以这里采用组合 trait。
///
/// 显式追加 `Send + Sync`：store 会作为 `AstroImageReader` 的字段，并在
/// `read_batch` 内通过 `rayon::par_iter` 跨线程共享访问；同时 `#[pyclass]`
/// 也要求结构体所有字段满足 `Send`。`FilesystemStore` 和经
/// `AsyncToSyncStorageAdapter` 包装后的 `AsyncOpendalStore` 均满足该 bound。
type ReadableStore = Arc<dyn ReadableListableStorageTraits + Send + Sync>;

/// 防御性 store wrapper，修复部分 S3 兼容存储（如阿里云 OSS）在 ListObjectsV2
/// 时返回根自身作为 common_prefix 导致 zarrs_opendal 的 `StorePrefix::try_from("/")` 失败的问题。
///
/// 仅重写 `list_dir`：使用 opendal `Operator` 直接列出，过滤掉 `path == "/"` 的条目。
/// 其他方法全部委托给内层 store。
struct OssStoreWrapper {
    inner: Arc<dyn ReadableListableStorageTraits + Send + Sync>,
    operator: Operator,
    runtime: &'static Runtime,
}

impl ReadableStorageTraits for OssStoreWrapper {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.inner.get(key)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.inner.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

impl ListableStorageTraits for OssStoreWrapper {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.inner.list()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.inner.list_prefix(prefix)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let entries = self
            .runtime
            .block_on(async {
                self.operator
                    .list_with(prefix.as_str())
                    .recursive(false)
                    .await
            })
            .map_err(|e| StorageError::Other(e.to_string()))?;

        let mut prefixes = Vec::new();
        let mut keys = Vec::new();

        for entry in &entries {
            let path = entry.path();
            // 防御性过滤：跳过根自身条目（部分 S3 兼容存储如 OSS 会返回根自身 "/"）
            if path == "/" {
                continue;
            }
            match entry.metadata().mode() {
                opendal::EntryMode::FILE => {
                    if let Ok(key) = StoreKey::try_from(path) {
                        keys.push(key);
                    }
                }
                opendal::EntryMode::DIR => {
                    if let Ok(prefix_entry) = StorePrefix::try_from(path) {
                        if &prefix_entry != prefix {
                            prefixes.push(prefix_entry);
                        }
                    }
                }
                opendal::EntryMode::Unknown => {}
            }
        }

        keys.sort();
        prefixes.sort();
        Ok(StoreKeysPrefixes::new(keys, prefixes))
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.inner.size_prefix(prefix)
    }

    fn size(&self) -> Result<u64, StorageError> {
        self.inner.size()
    }
}

// OssStoreWrapper 实现了 ReadableStorageTraits + ListableStorageTraits + 'static，
// 因此 zarrs_storage 中的 blanket impl 自动为它提供 ReadableListableStorageTraits 实现。
// 所有字段（Arc<dyn ...>, Operator, &'static Runtime）均为 Send + Sync，无需 unsafe impl。

/// 根据 path 字符串构建一个 zarrs 同步可读 store。
///
/// 支持：
/// - 本地路径（无 scheme 或 `file://`）：返回 `FilesystemStore`。
/// - `oss://[user[:pass]@]<bucket-or-host>/<path>`：返回经 AsyncToSyncStorageAdapter 包装的
///   `AsyncOpendalStore`，凭证、endpoint 与 [create_zarr_store] 的 Python 实现保持一致。
fn build_store(path: &str) -> PyResult<ReadableStore> {
    // 先识别是否为 OSS URL；非 URL 形式（绝对/相对本地路径）走 FilesystemStore。
    let parsed = match Url::parse(path) {
        Ok(u) => u,
        Err(ParseError::RelativeUrlWithoutBase) => match Url::parse(&format!("file://{}", path)) {
            Ok(u) => u,
            Err(e) => {
                return Err(PyValueError::new_err(format!(
                    "Invalid URL: {path:?}. Reason: {e}"
                )));
            }
        },
        Err(e) => {
            return Err(PyValueError::new_err(format!(
                "Invalid URL: {path:?}. Reason: {e}"
            )));
        }
    };

    let scheme = parsed.scheme();
    match scheme {
        "file" => {
            let local_path = parsed.path();
            let store = FilesystemStore::new(local_path).map_err(|e| {
                PyValueError::new_err(format!("Failed to open filesystem store: {}", e))
            })?;
            Ok(Arc::new(store))
        }
        "oss" | "http" | "https" => {
            // 与 Python 端 create_zarr_store 的语义对齐：
            //   - bucket 取 URL path 的第一段；
            //   - endpoint 优先用 URL host（`<scheme>://<host>[:port]`，oss scheme 默认使用 http），
            //     host 为空时回退到 OSS_ENDPOINT 环境变量，再回退到默认 endpoint；
            //   - access_key_id / access_key_secret 先读 URL userinfo，再回退到环境变量。
            let full_path = parsed.path().trim_start_matches('/');
            let (bucket, root_in_bucket) = match full_path.split_once('/') {
                Some((b, rest)) if !b.is_empty() => (b, rest),
                _ if !full_path.is_empty() => (full_path, ""),
                _ => {
                    return Err(PyValueError::new_err(
                        "oss:// URL missing bucket in path (expected oss://[host]/<bucket>/<path>)",
                    ));
                }
            };

            let access_key_id = if !parsed.username().is_empty() {
                Some(parsed.username().to_string())
            } else {
                env::var("OSS_ACCESS_KEY_ID").ok()
            };
            let access_key_secret = parsed
                .password()
                .map(|s| s.to_string())
                .or_else(|| env::var("OSS_ACCESS_KEY_SECRET").ok());

            let endpoint = match parsed.host_str() {
                Some(host) if !host.is_empty() => {
                    // oss:// 视为未指定具体协议，默认用 http；非 oss scheme 则沿用其 scheme。
                    let scheme = if scheme == "oss" { "http" } else { scheme };
                    if let Some(port) = parsed.port() {
                        format!("{}://{}:{}", scheme, host, port)
                    } else {
                        format!("{}://{}", scheme, host)
                    }
                }
                _ => env::var("OSS_ENDPOINT").unwrap_or_else(|_| DEFAULT_OSS_ENDPOINT.to_string()),
            };

            let root = format!("/{}", root_in_bucket);
            // 与 Python 端对齐：Python 通过 s3fs + endpoint_url 访问 OSS/内部存储，走的是 S3 兼容协议。
            // opendal 的 services-oss 仅适用于阿里云 OSS 原生协议，对 S3 兼容 endpoint 请求
            // 会成功建立连接但返回空数据。因此统一用 services-s3。
            let mut builder = S3::default()
                .bucket(bucket)
                .endpoint(&endpoint)
                .root(&root)
                // opendal S3 要求 region 字段。对于非 AWS 的 S3 兼容存储，region 仅用于签名，
                // 具体值无实际意义。优先取 AWS_REGION，否则默认 "auto"。
                .region(&env::var("OSS_REGION").unwrap_or_else(|_| "auto".to_string()));
            if let Some(k) = access_key_id.as_deref() {
                builder = builder.access_key_id(k);
            }
            if let Some(s) = access_key_secret.as_deref() {
                builder = builder.secret_access_key(s);
            }

            let op = Operator::new(builder)
                .map_err(|e| PyValueError::new_err(format!("Failed to build S3 operator: {}", e)))?
                .finish();

            // 凭证预检：验证连接和权限，避免后续 zarrs 操作因凭证错误返回空结果
            // 而报出令人困惑的 "group metadata is missing"。
            let rt = runtime();
            let check = rt.block_on(async { op.list_with("/").recursive(false).await });
            match check {
                Err(e) if e.kind() == opendal::ErrorKind::PermissionDenied => {
                    return Err(PyValueError::new_err(format!(
                        "OSS access denied: {}. \
                         Check OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET environment variables \
                         or credentials in the URL (oss://key:secret@host/bucket/path)",
                        e
                    )));
                }
                Err(e) if e.kind() == opendal::ErrorKind::NotFound => {
                    return Err(PyValueError::new_err(format!(
                        "OSS bucket or path not found: {}. \
                         Check the bucket name and path in the URL",
                        e
                    )));
                }
                Err(e) => {
                    return Err(PyValueError::new_err(format!(
                        "OSS connection failed: {}. \
                         Check endpoint, credentials, and network connectivity",
                        e
                    )));
                }
                _ => {}
            }

            let async_store = Arc::new(AsyncOpendalStore::new(op.clone()));
            let sync_store = AsyncToSyncStorageAdapter::new(async_store, TokioBlockOn);
            let wrapper = OssStoreWrapper {
                inner: Arc::new(sync_store),
                operator: op,
                runtime: runtime(),
            };
            Ok(Arc::new(wrapper))
        }
        other => Err(PyValueError::new_err(format!(
            "unsupported scheme {:?} for AstroImageReader path",
            other
        ))),
    }
}

/// 存储每个 interval 的元数据: (subset_path, offset)
#[derive(Clone, Debug, Eq, PartialEq)]
struct IntervalData {
    subset_path: String,
    offset: usize,
}

type IvType = Interval<usize, IntervalData>;

/// 多巡天图像读取器
///
/// 使用 rust-lapper 实现的 interval tree 进行索引，
/// 使用 zarrs 库高效读取 zarr 格式的天文图像数据。
#[pyclass]
pub struct AstroImageReader {
    // /// zarr 数据集根路径（本地路径或 oss:// URL，原样保存）。
    // zarr_root_path: String,
    /// 共享的 zarrs 同步 store。构造时建立一次，全生命周期复用，避免每次
    /// `read_single_example` 都重建 `FilesystemStore` / opendal `Operator`
    /// 及其底层 HTTP 连接池。
    store: ReadableStore,
    /// interval tree 索引
    index: Lapper<usize, IntervalData>,
    /// 子集列表 (用于遍历)
    subsets: Vec<String>,
    /// 裁剪大小
    crop_size: usize,
    /// 是否禁用 mask
    disable_mask: bool,
    /// 是否禁用 ivar
    disable_ivar: bool,
    /// 通道数
    num_channels: usize,
    /// 总样本数
    total_samples: usize,
    /// 块读取粒度: 每次 I/O 读取的连续样本数。
    /// 当 > 1 时，`read_batch` 接收的每个 index 代表一个 block 的起始全局索引，
    /// 单次 `retrieve_array_subset` 覆盖 [start..start+read_block_size] 的 N 范围。
    read_block_size: usize,
}

#[pymethods]
impl AstroImageReader {
    /// 创建新的 AstroImageReader 实例
    ///
    /// Args:
    ///     zarr_root_path: zarr 数据集的根路径，支持本地路径和 `oss://` URL。
    ///     crop_size: 裁剪大小 (默认 96)
    ///     disable_mask: 是否禁用 mask 读取 (默认 False)
    ///     disable_ivar: 是否禁用 ivar 读取 (默认 False)
    ///     max_chunk_size: 最大切片大小，用于将大索引区间切分 (默认 5000)
    #[new]
    #[pyo3(signature = (zarr_root_path, crop_size=96, disable_mask=false, disable_ivar=false, max_chunk_size=5000, read_block_size=1))]
    pub fn new(
        py: Python<'_>,
        zarr_root_path: &str,
        crop_size: usize,
        disable_mask: bool,
        disable_ivar: bool,
        max_chunk_size: usize,
        read_block_size: usize,
    ) -> PyResult<Self> {
        if read_block_size == 0 {
            return Err(PyValueError::new_err("read_block_size must be >= 1"));
        }

        let zarr_root_path = zarr_root_path.trim_end_matches('/').to_string();

        // 构建索引涉及远程/本地 I/O，释放 GIL 以允许其他 Python 线程并发执行。
        // 同时把 build_index 内部已创建的 store 取出复用，避免后续读取时重建。
        let (store, index, subsets, total_samples, num_channels) =
            py.detach(|| Self::build_index(&zarr_root_path, max_chunk_size))?;

        Ok(AstroImageReader {
            store,
            index,
            subsets,
            crop_size,
            disable_mask,
            disable_ivar,
            num_channels,
            total_samples,
            read_block_size,
        })
    }

    /// 获取总样本数
    #[getter]
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// 获取通道数
    #[getter]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// 获取裁剪大小
    #[getter]
    pub fn crop_size(&self) -> usize {
        self.crop_size
    }

    /// 获取是否禁用 mask
    #[getter]
    pub fn disable_mask(&self) -> bool {
        self.disable_mask
    }

    /// 获取是否禁用 ivar
    #[getter]
    pub fn disable_ivar(&self) -> bool {
        self.disable_ivar
    }

    /// 获取子集列表
    #[getter]
    pub fn subsets(&self) -> Vec<String> {
        self.subsets.clone()
    }

    /// 批量读取样本
    ///
    /// 当 `read_block_size > 1` 时，indices 中每个值代表一个 block 的起始全局索引，
    /// 实际输出的 batch_size = indices.len() * read_block_size。
    /// 当 `read_block_size == 1` 时，行为与传统逐样本读取等价。
    ///
    /// Args:
    ///     indices: block 起始索引数组 (numpy int64 array)
    ///
    /// Returns:
    ///     tuple: (flux, mask, ivar)
    ///         - flux: shape (N, C, H, W) 的 float32 数组
    ///         - mask: shape (N, C, H, W) 或 (N, 1, H, W) 的 bool 数组，未禁用时返回，否则为 None
    ///         - ivar: shape (N, C, H, W) 的 float32 数组，未禁用时返回，否则为 None
    ///
    /// 注意: 调用期间内部会释放 GIL，调用方必须保证在本次调用返回前不从其他
    /// 线程修改 `indices` 数组，否则行为未定义。
    #[pyo3(signature = (indices: "np.ndarray")
        -> "tuple[np.ndarray, \
                  np.ndarray | None, \
                  np.ndarray | None] \
            | None")]
    pub fn read_batch<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'py, usize>,
    ) -> Example<'py> {
        let idx_slice: &[usize] = indices.as_slice()?;
        let num_blocks = idx_slice.len();

        // 并行 I/O + ndarray::concatenate 均为纯 Rust 工作，无需 GIL。
        let (flux_4d, mask_4d, ivar_4d) =
            py.detach(|| -> PyResult<_> {
                // 并行读取所有 block，collect 保序 + 短路错误。
                let (flux_blocks, mask_blocks, ivar_blocks) = idx_slice
                    .iter()
                    .map(|&block_start| {
                        let (subset_path, local_idx) = self
                            .get_example_addr(block_start)
                            .ok_or_else(|| PyErr::from(MsirError::IndexOutOfBounds(block_start)))?;
                        self.read_block_examples(subset_path, local_idx)
                    })
                    .collect::<PyResult<Vec<_>>>()?
                    .into_iter()
                    .fold(
                        (
                            Vec::with_capacity(num_blocks),
                            if self.disable_mask {
                                Vec::new()
                            } else {
                                Vec::with_capacity(num_blocks)
                            },
                            if self.disable_ivar {
                                Vec::new()
                            } else {
                                Vec::with_capacity(num_blocks)
                            },
                        ),
                        |(mut fs, mut ms, mut is), (f, m, i)| {
                            fs.push(f);
                            if let Some(m) = m {
                                ms.push(m);
                            }
                            if let Some(i) = i {
                                is.push(i);
                            }
                            (fs, ms, is)
                        },
                    );

                // 沿 axis 0 拼接所有 block
                let flux_views: Vec<_> = flux_blocks.iter().map(|f| f.view()).collect();
                let flux_4d = ndarray::concatenate(Axis(0), &flux_views)
                    .map_err(|e| PyValueError::new_err(format!("Concatenate error: {}", e)))?;

                let mask_4d = if !self.disable_mask {
                    let mask_views: Vec<_> =
                        mask_blocks.iter().map(|m| m.view()).collect::<Vec<_>>();

                    if mask_views.len() == flux_blocks.len() {
                        Some(ndarray::concatenate(Axis(0), &mask_views).map_err(|e| {
                            PyValueError::new_err(format!("Concatenate error: {}", e))
                        })?)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let ivar_4d = if !self.disable_ivar {
                    let ivar_views: Vec<_> =
                        ivar_blocks.iter().map(|i| i.view()).collect::<Vec<_>>();
                    if ivar_views.len() == flux_blocks.len() {
                        Some(ndarray::concatenate(Axis(0), &ivar_views).map_err(|e| {
                            PyValueError::new_err(format!("Concatenate error: {}", e))
                        })?)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let flux_shape = flux_4d.shape();
                if flux_shape.len() != 4 {
                    return Err(PyValueError::new_err("Flux must be 4-dimensional"));
                }

                if let Some(mask_4d_) = &mask_4d {
                    let mask_shape = mask_4d_.shape();
                    if mask_shape.len() != 4 {
                        return Err(PyValueError::new_err("Mask must be 4-dimensional"));
                    }

                    let batch_size = flux_shape[0];
                    let num_channels = flux_shape[1];
                    let height = flux_shape[2];
                    let width = flux_shape[3];

                    if mask_shape[0] != batch_size
                        || (mask_shape[1] != num_channels && mask_shape[1] != 1)
                        || mask_shape[2] != height
                        || mask_shape[3] != width
                    {
                        return Err(PyValueError::new_err(format!(
                            "Mask shape ({mask_shape:?}) mismatch with flux ({flux_shape:?})"
                        )));
                    }
                }

                if let Some(ivar_4d_) = &ivar_4d {
                    let ivar_shape = ivar_4d_.shape();

                    if ivar_shape != flux_shape {
                        return Err(PyValueError::new_err(format!(
                            "IVar shape ({ivar_shape:?}) mismatch with flux ({flux_shape:?})"
                        )));
                    }
                }

                Ok((flux_4d, mask_4d, ivar_4d))
            })?;

        // 转换为 PyArray (零拷贝)
        Ok((
            flux_4d.into_pyarray(py),
            mask_4d.map(|m| m.into_pyarray(py)),
            ivar_4d.map(|i| i.into_pyarray(py)),
        ))
    }

    /// 读取单个样本 (返回 numpy 数组)
    ///
    /// 注意: 调用期间内部会释放 GIL。
    #[pyo3(signature = (index: "np.uintp | int")
        -> "tuple[np.ndarray, \
                  np.ndarray | None, \
                  np.ndarray | None] \
            | None")]
    pub fn read_example<'py>(&self, py: Python<'py>, index: usize) -> Example<'py> {
        // zarr 读取为纯 Rust I/O，释放 GIL；insert_axis 属于零成本视图操作，顺带一起放入闭包。
        let (flux_4d, mask_4d, ivar_4d) = py.detach(|| -> PyResult<_> {
            let addr = self
                .get_example_addr(index)
                .ok_or(MsirError::IndexOutOfBounds(index))?;
            let (flux, mask, ivar) = self.read_single_example(addr.0, addr.1)?;
            Ok((
                flux.insert_axis(Axis(0)),
                mask.map(|m| m.insert_axis(Axis(0))),
                ivar.map(|i| i.insert_axis(Axis(0))),
            ))
        })?;

        Ok((
            flux_4d.into_pyarray(py),
            mask_4d.map(|m| m.into_pyarray(py)),
            ivar_4d.map(|i| i.into_pyarray(py)),
        ))
    }

    /// 获取样本的地址 (subset_path, local_idx)
    #[inline]
    pub fn get_addr(&self, index: usize) -> Option<(String, usize)> {
        self.get_example_addr(index)
            .map(|(subset, local_idx)| (subset.to_owned(), local_idx))
    }

    /// 按 block 粒度收集样本 ID。
    ///
    /// 每个返回的 ID 代表连续 `block_size` 个样本的起始全局索引。
    /// 子集尾部不足 `block_size` 的样本将被丢弃。
    ///
    /// Args:
    ///     rank: 进程 rank
    ///     world_size: 进程总数
    ///     block_size: 块大小 (= read_block_size)
    ///
    /// Returns:
    ///     list[range]: 每个元素为 block 起始索引的 range
    pub fn collect_block_ids<'py>(
        &self,
        py: Python<'py>,
        rank: usize,
        world_size: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let block_size = self.read_block_size;
        let builtins = py.import("builtins")?;
        let range_type = builtins.getattr("range")?;
        let list = PyList::empty(py);
        for iv in self.index.iter() {
            let start = iv.start;
            let end = iv.stop;
            let num_samples = end - start;
            let num_blocks = num_samples / block_size;
            if num_blocks == 0 {
                continue;
            }
            let aligned_end = start + num_blocks * block_size;
            // 以 block 为单位交错分配给各 rank，步长为 world_size * block_size
            let shard_start = start + rank * block_size;
            if shard_start < aligned_end {
                let step = world_size * block_size;
                let r = range_type.call1((shard_start as i64, aligned_end as i64, step as i64))?;
                list.append(r)?;
            }
        }
        Ok(list)
    }

    /// 估算分片后的批次数量（block 粒度版本）
    ///
    /// 遍历所有 rank，计算每个 rank 分到的 block 数，取最小值后除以
    /// 每 batch 需要的 block 数 (= batch_size / block_size)，保证所有
    /// rank 的 batch 数一致，避免 DDP 死锁。
    ///
    /// 当 `block_size == 1` 时退化为逐样本的 estimate。
    ///
    /// Args:
    ///     batch_size: 批次大小
    ///     world_size: 总进程数
    ///     block_size: 块大小 (= read_block_size)
    ///
    /// Returns:
    ///     int: 所有 rank 中最小的批次数量
    pub fn estimate_sharded_batches(&self, batch_size: usize, world_size: usize) -> usize {
        let block_size = self.read_block_size;
        let blocks_per_batch = batch_size / block_size;
        if blocks_per_batch == 0 {
            return 0;
        }
        let mut sizes = vec![0usize; world_size];

        for (rank, rank_size_ref) in sizes.iter_mut().enumerate() {
            for iv in self.index.iter() {
                let start = iv.start;
                let end = iv.stop;
                let num_samples = end - start;
                let num_blocks = num_samples / block_size;
                if num_blocks == 0 {
                    continue;
                }
                let aligned_end = start + num_blocks * block_size;
                let shard_start = start + rank * block_size;
                if shard_start >= aligned_end {
                    continue;
                }
                // range(shard_start, aligned_end, world_size * block_size) 的长度
                let step = world_size * block_size;
                let rank_blocks = (aligned_end - shard_start).div_ceil(step);
                *rank_size_ref += rank_blocks;
            }
        }

        let min_blocks = sizes.iter().copied().min().unwrap_or(0);
        min_blocks / blocks_per_batch
    }
}

impl AstroImageReader {
    /// 构建索引
    fn build_index(zarr_root_path: &str, max_chunk_size: usize) -> IndexType {
        let store = build_store(zarr_root_path)?;
        let root = Group::open(store.clone(), "/")
            .map_err(|e| PyValueError::new_err(format!("Failed to open root group: {}", e)))?;

        let child_groups = root
            .child_groups()
            .map_err(|e| PyValueError::new_err(format!("Failed to list children: {}", e)))?;

        let mut intervals: Vec<IvType> = Vec::new();
        let mut subsets: Vec<String> = Vec::new();
        let mut start_idx: usize = 0;
        let mut num_channels: usize = 0;

        // 遍历所有子集
        for child_group in child_groups {
            // path() 返回 NodePath，可以转换为 String
            let full_path = child_group.path().to_string();
            // 获取最后一个路径组件作为 subset 名称
            let subset_name = full_path
                .trim_start_matches('/')
                .split('/')
                .next_back()
                .unwrap_or("")
                .to_string();
            if subset_name.is_empty() {
                continue;
            }
            let subset_path = format!("/{}", subset_name);
            // 获取 flux 数组的形状
            let flux_array = Array::open(store.clone(), &format!("{}/flux", subset_path))
                .map_err(|e| PyValueError::new_err(format!("Failed to open flux array: {}", e)))?;

            let shape = flux_array.shape();
            if shape.len() != 4 {
                return Err(PyValueError::new_err(format!(
                    "Expected flux array to have at least 2 dimensions, got {}",
                    shape.len()
                )));
            }
            let num_examples = shape[0] as usize;

            if num_channels == 0 {
                num_channels = shape[1] as usize;
            }

            subsets.push(subset_name.clone());

            // 切分为多个小区间
            if max_chunk_size == 0 || num_examples <= max_chunk_size {
                intervals.push(Interval {
                    start: start_idx,
                    stop: start_idx + num_examples,
                    val: IntervalData {
                        subset_path: subset_name.clone(),
                        offset: 0,
                    },
                });
                start_idx += num_examples;
            } else {
                let mut offset = 0;
                let mut remaining = num_examples;
                while remaining > 0 {
                    let chunk_size = remaining.min(max_chunk_size);
                    intervals.push(Interval {
                        start: start_idx,
                        stop: start_idx + chunk_size,
                        val: IntervalData {
                            subset_path: subset_name.clone(),
                            offset,
                        },
                    });
                    start_idx += chunk_size;
                    offset += chunk_size;
                    remaining -= chunk_size;
                }
            }
        }

        let total_samples = start_idx;
        let lapper = Lapper::new(intervals);

        Ok((store, lapper, subsets, total_samples, num_channels))
    }

    /// 获取样本地址
    fn get_example_addr(&self, example_id: usize) -> Option<(&str, usize)> {
        let mut results = self.index.find(example_id, example_id + 1);

        results.next().map(|interval| {
            let local_idx = example_id - interval.start + interval.val.offset;
            (interval.val.subset_path.as_ref(), local_idx)
        })
    }

    /// 读取单个样本
    fn read_single_example(&self, subset_path: &str, local_idx: usize) -> ExampleArray3 {
        // 直接复用构造时缓存的 store；clone 仅递增 Arc 引用计数，零开销。
        let store = self.store.clone();

        let flux_path = format!("/{}/flux", subset_path);
        let flux_array = Array::open(store.clone(), &flux_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to open flux: {}", e)))?;

        // 计算中心裁剪的范围
        let shape = flux_array.shape();
        if shape.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "Expected flux array to have 4 dimensions (N, C, H, W), got {}",
                shape.len()
            )));
        }
        let height = shape[2] as usize;
        let width = shape[3] as usize;
        if height < self.crop_size || width < self.crop_size {
            return Err(PyValueError::new_err(format!(
                "Image size ({}x{}) is smaller than crop_size ({})",
                height, width, self.crop_size
            )));
        }
        let start_y = (height - self.crop_size) / 2;
        let start_x = (width - self.crop_size) / 2;

        // 读取 flux
        let flux_subset = ArraySubset::new_with_ranges(&[
            local_idx as u64..(local_idx + 1) as u64,
            0..shape[1],
            start_y as u64..(start_y + self.crop_size) as u64,
            start_x as u64..(start_x + self.crop_size) as u64,
        ]);

        let flux_data: ndarray::ArrayD<f32> = flux_array
            .retrieve_array_subset_ndarray(&flux_subset)
            .map_err(|e| PyValueError::new_err(format!("Failed to read flux: {}", e)))?;

        // 移除 batch 维度: (1, C, H, W) -> (C, H, W)
        let flux_3d = flux_data
            .into_shape_with_order(ndarray::IxDyn(&[
                self.num_channels,
                self.crop_size,
                self.crop_size,
            ]))
            .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| PyValueError::new_err(format!("Dimension error: {}", e)))?;

        // 读取 mask（未禁用时）
        // mask 形状可能与 flux 同（4D）或缺通道维（3D）
        let mask_3d = if !self.disable_mask {
            let mask_path = format!("/{}/mask", subset_path);
            let mask_array = Array::open(store.clone(), &mask_path)
                .map_err(|e| PyValueError::new_err(format!("Failed to open mask: {}", e)))?;

            let mask_shape = mask_array.shape();
            let mask_subset = if mask_shape.len() == 4 {
                ArraySubset::new_with_ranges(&[
                    local_idx as u64..(local_idx + 1) as u64,
                    0..mask_shape[1],
                    start_y as u64..(start_y + self.crop_size) as u64,
                    start_x as u64..(start_x + self.crop_size) as u64,
                ])
            } else {
                ArraySubset::new_with_ranges(&[
                    local_idx as u64..(local_idx + 1) as u64,
                    start_y as u64..(start_y + self.crop_size) as u64,
                    start_x as u64..(start_x + self.crop_size) as u64,
                ])
            };

            let mask_data: ndarray::ArrayD<bool> = mask_array
                .retrieve_array_subset_ndarray(&mask_subset)
                .map_err(|e| PyValueError::new_err(format!("Failed to read mask: {}", e)))?;

            let mask_3d = if mask_shape.len() == 4 {
                let num_mask_channels = mask_shape[1] as usize;
                mask_data
                    .into_shape_with_order(ndarray::IxDyn(&[
                        num_mask_channels,
                        self.crop_size,
                        self.crop_size,
                    ]))
                    .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?
                    .into_dimensionality::<ndarray::Ix3>()
                    .map_err(|e| PyValueError::new_err(format!("Dimension error: {}", e)))?
            } else {
                mask_data
                    .into_shape_with_order(ndarray::IxDyn(&[1, self.crop_size, self.crop_size]))
                    .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?
                    .into_dimensionality::<ndarray::Ix3>()
                    .map_err(|e| PyValueError::new_err(format!("Dimension error: {}", e)))?
            };

            Some(mask_3d)
        } else {
            None
        };

        // 读取 ivar（未禁用时）
        let ivar_3d = if !self.disable_ivar {
            let ivar_path = format!("/{}/ivar", subset_path);
            let ivar_array = Array::open(store.clone(), &ivar_path)
                .map_err(|e| PyValueError::new_err(format!("Failed to open ivar: {}", e)))?;

            let ivar_subset = ArraySubset::new_with_ranges(&[
                local_idx as u64..(local_idx + 1) as u64,
                0..shape[1],
                start_y as u64..(start_y + self.crop_size) as u64,
                start_x as u64..(start_x + self.crop_size) as u64,
            ]);

            let ivar_data: ndarray::ArrayD<f32> = ivar_array
                .retrieve_array_subset_ndarray(&ivar_subset)
                .map_err(|e| PyValueError::new_err(format!("Failed to read ivar: {}", e)))?;

            let ivar_3d = ivar_data
                .into_shape_with_order(ndarray::IxDyn(&[
                    self.num_channels,
                    self.crop_size,
                    self.crop_size,
                ]))
                .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| PyValueError::new_err(format!("Dimension error: {}", e)))?;

            Some(ivar_3d)
        } else {
            None
        };

        Ok((flux_3d, mask_3d, ivar_3d))
    }

    /// 一次性读取同一 subset 内连续 `block_size` 个样本。
    ///
    /// 与 `read_single_example` 的关键区别:
    /// - `ArraySubset` 的 N 维范围为 `[start..start+block_size]`
    /// - 返回 4D 数组 (block_size, C, H, W)
    /// - 单次 `retrieve_array_subset` 调用覆盖整个 block
    fn read_block_examples(&self, subset_path: &str, start_local_idx: usize) -> BlockResult {
        let store = &self.store;
        let block_size = self.read_block_size;

        let flux_path = format!("/{}/flux", subset_path);
        let flux_array = Array::open(store.clone(), &flux_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to open flux: {}", e)))?;

        // 计算中心裁剪范围
        let shape = flux_array.shape();
        if shape.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "Expected flux array to have 4 dimensions (N, C, H, W), got {}",
                shape.len()
            )));
        }
        let height = shape[2] as usize;
        let width = shape[3] as usize;
        if height < self.crop_size || width < self.crop_size {
            return Err(PyValueError::new_err(format!(
                "Image size ({}x{}) is smaller than crop_size ({})",
                height, width, self.crop_size
            )));
        }
        let start_y = (height - self.crop_size) / 2;
        let start_x = (width - self.crop_size) / 2;

        // 读取 flux: (block_size, C, H, W)
        let flux_subset = ArraySubset::new_with_ranges(&[
            start_local_idx as u64..(start_local_idx + block_size) as u64,
            0..shape[1],
            start_y as u64..(start_y + self.crop_size) as u64,
            start_x as u64..(start_x + self.crop_size) as u64,
        ]);

        let flux_data: ndarray::ArrayD<f32> = flux_array
            .retrieve_array_subset_ndarray(&flux_subset)
            .map_err(|e| PyValueError::new_err(format!("Failed to read flux: {}", e)))?;

        let flux_4d = flux_data
            .into_shape_with_order(ndarray::IxDyn(&[
                block_size,
                self.num_channels,
                self.crop_size,
                self.crop_size,
            ]))
            .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|e| PyValueError::new_err(format!("Dimension error: {}", e)))?;

        // 读取 mask
        let mask_4d = if !self.disable_mask {
            let mask_path = format!("/{}/mask", subset_path);
            let mask_array = Array::open(store.clone(), &mask_path)
                .map_err(|e| PyValueError::new_err(format!("Failed to open mask: {}", e)))?;

            let mask_shape = mask_array.shape();
            let (mask_subset, num_mask_channels) = if mask_shape.len() == 4 {
                (
                    ArraySubset::new_with_ranges(&[
                        start_local_idx as u64..(start_local_idx + block_size) as u64,
                        0..mask_shape[1],
                        start_y as u64..(start_y + self.crop_size) as u64,
                        start_x as u64..(start_x + self.crop_size) as u64,
                    ]),
                    mask_shape[1] as usize,
                )
            } else {
                (
                    ArraySubset::new_with_ranges(&[
                        start_local_idx as u64..(start_local_idx + block_size) as u64,
                        start_y as u64..(start_y + self.crop_size) as u64,
                        start_x as u64..(start_x + self.crop_size) as u64,
                    ]),
                    1,
                )
            };

            let mask_data: ndarray::ArrayD<bool> = mask_array
                .retrieve_array_subset_ndarray(&mask_subset)
                .map_err(|e| PyValueError::new_err(format!("Failed to read mask: {}", e)))?;

            let mask_4d = mask_data
                .into_shape_with_order(ndarray::IxDyn(&[
                    block_size,
                    num_mask_channels,
                    self.crop_size,
                    self.crop_size,
                ]))
                .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| PyValueError::new_err(format!("Dimension error: {}", e)))?;

            Some(mask_4d)
        } else {
            None
        };

        // 读取 ivar
        let ivar_4d = if !self.disable_ivar {
            let ivar_path = format!("/{}/ivar", subset_path);
            let ivar_array = Array::open(store.clone(), &ivar_path)
                .map_err(|e| PyValueError::new_err(format!("Failed to open ivar: {}", e)))?;

            let ivar_subset = ArraySubset::new_with_ranges(&[
                start_local_idx as u64..(start_local_idx + block_size) as u64,
                0..shape[1],
                start_y as u64..(start_y + self.crop_size) as u64,
                start_x as u64..(start_x + self.crop_size) as u64,
            ]);

            let ivar_data: ndarray::ArrayD<f32> = ivar_array
                .retrieve_array_subset_ndarray(&ivar_subset)
                .map_err(|e| PyValueError::new_err(format!("Failed to read ivar: {}", e)))?;

            let ivar_4d = ivar_data
                .into_shape_with_order(ndarray::IxDyn(&[
                    block_size,
                    self.num_channels,
                    self.crop_size,
                    self.crop_size,
                ]))
                .map_err(|e| PyValueError::new_err(format!("Reshape error: {}", e)))?
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| PyValueError::new_err(format!("Dimension error: {}", e)))?;

            Some(ivar_4d)
        } else {
            None
        };

        Ok((flux_4d, mask_4d, ivar_4d))
    }
}

// Python 模块定义。
//
// 使用声明式模块语法（`#[pymodule] mod ...`），以便通过 `#[pymodule_init]`
// 在模块导入阶段完成 tokio runtime 的一次性初始化。
#[pymodule]
mod msir {
    use pyo3::exceptions::PyImportError;
    use pyo3::prelude::*;
    use tokio::runtime::Runtime;

    // 将 `AstroImageReader` 暴露到 Python 模块命名空间。
    #[pymodule_export]
    use super::AstroImageReader;

    /// 配置 zarrs 内部并发参数。
    ///
    /// Args:
    ///     chunk_concurrent_minimum: chunk 级最小并发数。为 None 时保持不变。
    ///     codec_concurrent_target: codec 编解码总并发上限。为 None 时保持不变。
    #[pyfunction]
    #[pyo3(signature = (chunk_concurrent_minimum=None, codec_concurrent_target=None))]
    fn configure(chunk_concurrent_minimum: Option<usize>, codec_concurrent_target: Option<usize>) {
        use zarrs::config::global_config_mut;

        let mut cfg = global_config_mut();
        if let Some(v) = chunk_concurrent_minimum {
            cfg.set_chunk_concurrent_minimum(v);
            log::debug!(target: "msir", "zarrs.chunk_concurrent_minimum set to {v}");
        }
        if let Some(v) = codec_concurrent_target {
            cfg.set_codec_concurrent_target(v);
            log::debug!(target: "msir", "zarrs.codec_concurrent_target set to {v}");
        }
    }

    /// 模块初始化：在此处创建 tokio runtime。
    ///
    /// - 初始化只发生一次，且发生在 Python `import msir` 的过程中；
    /// - 若创建失败，直接抛出 `ImportError`，让 `import` 自身失败，
    ///   避免后续运行时再处理 runtime 不可用的情况；
    /// - 通过 `OnceLock::set`（而非 `get_or_init`）写入，后续访问只读，
    ///   不存在 GIL + OnceLock 初始化闭包带来的死锁风险
    ///   （见 PyO3 FAQ）。
    #[pymodule_init]
    fn init_module(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        // 使用 env_logger 让本扩展库的日志有一个专属开关，不依赖
        // 全局 RUST_LOG，也不引入 pyo3-log。Python 进程设置环境变量
        // `MSIR_LOG`（如 `MSIR_LOG=debug`）后，msir 内部的 log 调用才会
        // 在 stderr 输出；未设置时默认为 `off`，什么也不打印。
        // `try_init` 重复调用会返回 Err，不应阻塞模块加载。
        let _ = env_logger::Builder::from_env(
            env_logger::Env::new()
                .filter_or("MSIR_LOG", "off")
                .write_style("MSIR_LOG_STYLE"),
        )
        .try_init();

        // 根据 LOCAL_WORLD_SIZE 环境变量按比例缩小 zarrs 的 codec 总并发上限，
        // 避免 DataLoader 多 worker 场景下各 worker 之和超出物理核心数。
        // 解析失败或值为 0 时回退到 1（不做缩放）。
        let local_world_size: usize = std::env::var("LOCAL_WORLD_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1);

        // 读取 zarrs 当前（即默认）codec_concurrent_target，再除以 LOCAL_WORLD_SIZE；
        // 至少保留 1 以避免被设置为 0（zarrs 中 0 表示不限制，并非我们想要的语义）。
        let default_target = rayon::current_num_threads();
        let scaled_target = (default_target / local_world_size).max(1);
        log::debug!(
            target: "msir",
            "init: LOCAL_WORLD_SIZE={}, zarrs default codec_concurrent_target={}, scaled to {}",
            local_world_size,
            default_target,
            scaled_target,
        );

        configure(Some(2), Some(scaled_target));
        let rt = Runtime::new().map_err(|e| {
            PyImportError::new_err(format!("failed to create tokio runtime for msir: {}", e))
        })?;
        super::RUNTIME
            .set(rt)
            .map_err(|_| PyImportError::new_err("msir tokio runtime is already initialized"))?;
        Ok(())
    }
}
