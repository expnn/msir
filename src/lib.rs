use ndarray::{Array3, Array4, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray4, PyReadonlyArray1};
use pyo3::types::PyList;
use opendal::{Operator, services::S3};
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
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
use zarrs::storage::ReadableListableStorageTraits;
use zarrs::storage::storage_adapter::async_to_sync::{AsyncToSyncBlockOn, AsyncToSyncStorageAdapter};
use zarrs_opendal::AsyncOpendalStore;

// noinspection HttpUrlsUsage
const DEFAULT_OSS_ENDPOINT: &str = "http://oss-cn-hangzhou-zjy-d01-a.res.cloud.zhejianglab.com";

#[derive(Error, Debug)]
pub enum MsirError {
    #[error("Zarr error: {0}")]
    Zarr(String),
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(i64),
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
type ReadableStore = Arc<dyn ReadableListableStorageTraits>;

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
        Err(ParseError::RelativeUrlWithoutBase) => {
            match Url::parse(&format!("file://{}", path)) {
                Ok(u) => u,
                Err(e) => return Err(PyValueError::new_err(format!("Invalid URL: {path:?}. Reason: {e}"))),
            }
        },
        Err(e) => return Err(PyValueError::new_err(format!("Invalid URL: {path:?}. Reason: {e}"))),
    };

    let scheme = parsed.scheme();
    match scheme {
        "file" => {
            let local_path = parsed.path();
            let store = FilesystemStore::new(local_path)
                .map_err(|e| PyValueError::new_err(format!("Failed to open filesystem store: {}", e)))?;
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
                .bucket(&bucket)
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
            // runtime 已在模块初始化阶段创建，这里无需再次校验。
            let async_store = Arc::new(AsyncOpendalStore::new(op));
            let sync_store = AsyncToSyncStorageAdapter::new(async_store, TokioBlockOn);
            Ok(Arc::new(sync_store))
        }
        other => Err(PyValueError::new_err(format!("unsupported scheme {:?} for AstroImageReader path", other))),
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
    /// zarr 数据集根路径（本地路径或 oss:// URL，原样保存）。
    zarr_root_path: String,
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
    #[pyo3(signature = (zarr_root_path, crop_size=96, disable_mask=false, disable_ivar=false, max_chunk_size=5000))]
    pub fn new(
        py: Python<'_>,
        zarr_root_path: &str,
        crop_size: usize,
        disable_mask: bool,
        disable_ivar: bool,
        max_chunk_size: usize,
    ) -> PyResult<Self> {
        let zarr_root_path = zarr_root_path.trim_end_matches('/').to_string();

        // 构建索引涉及远程/本地 I/O，释放 GIL 以允许其他 Python 线程并发执行。
        let (index, subsets, total_samples, num_channels) =
            py.detach(|| Self::build_index(&zarr_root_path, max_chunk_size))?;

        Ok(AstroImageReader {
            zarr_root_path,
            index,
            subsets,
            crop_size,
            disable_mask,
            disable_ivar,
            num_channels,
            total_samples,
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
    /// Args:
    ///     indices: 样本索引数组 (numpy int64 array)
    ///
    /// Returns:
    ///     tuple: (flux, mask, ivar)
    ///         - flux: shape (N, C, H, W) 的 float32 数组
    ///         - mask: shape (N, C, H, W) 或 (N, 1, H, W) 的 bool 数组，未禁用时返回，否则为 None
    ///         - ivar: shape (N, C, H, W) 的 float32 数组，未禁用时返回，否则为 None
    ///
    /// 注意: 调用期间内部会释放 GIL，调用方必须保证在本次调用返回前不从其他
    /// 线程修改 `indices` 数组，否则行为未定义。
    pub fn read_batch<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Option<Bound<'py, PyArray4<bool>>>,
        Option<Bound<'py, PyArray4<f32>>>,
    )> {
        // &[i64] 指向 numpy 底层 C 缓冲区，满足 Send + Sync + Ungil，可直接跨
        // detach 边界使用；调用方需遵守上方 doc 中“不得并发修改 indices”的约定。
        let idx_slice: &[i64] = indices.as_slice()?;

        // 并行 I/O + ndarray::stack 均为纯 Rust 工作，无需 GIL，整体放入 detach 闭包。
        let (flux_4d, mask_4d, ivar_4d) = py.detach(|| -> PyResult<(
            Array4<f32>,
            Option<Array4<bool>>,
            Option<Array4<f32>>,
        )> {
            // 并行读取所有样本：地址解析与 zarr 读取在同一条并行管线中完成，
            // 避免中间 Vec<Option<...>> 的一次性分配。
            let results: Vec<PyResult<(Array3<f32>, Option<Array3<bool>>, Option<Array3<f32>>)>> = idx_slice
                .par_iter()
                .map(|&idx| match self.get_example_addr(idx as usize) {
                    Some((subset_path, local_idx)) => self.read_single_example(&subset_path, local_idx),
                    None => Err(MsirError::IndexOutOfBounds(idx).into()),
                })
                .collect();

            // 收集有效结果
            let mut fluxes: Vec<Array3<f32>> = Vec::with_capacity(results.len());
            let mut masks: Vec<Array3<bool>> = Vec::with_capacity(results.len());
            let mut ivars: Vec<Array3<f32>> = Vec::with_capacity(results.len());

            for result in results {
                let (flux, mask, ivar) = result?;
                fluxes.push(flux);
                if let Some(m) = mask {
                    masks.push(m);
                }
                if let Some(i) = ivar {
                    ivars.push(i);
                }
            }

            if fluxes.is_empty() {
                return Err(PyValueError::new_err("No valid samples found"));
            }

            // 将结果堆叠为 4D 数组
            let flux_views: Vec<_> = fluxes.iter().map(|a| a.view()).collect();
            let flux_4d = ndarray::stack(Axis(0), &flux_views)
                .map_err(|e| PyValueError::new_err(format!("Stack error: {}", e)))?;

            let mask_4d = if !masks.is_empty() && masks.len() == fluxes.len() {
                let mask_views: Vec<_> = masks.iter().map(|a| a.view()).collect();
                Some(
                    ndarray::stack(Axis(0), &mask_views)
                        .map_err(|e| PyValueError::new_err(format!("Stack error: {}", e)))?,
                )
            } else {
                None
            };

            let ivar_4d = if !ivars.is_empty() && ivars.len() == fluxes.len() {
                let ivar_views: Vec<_> = ivars.iter().map(|a| a.view()).collect();
                Some(
                    ndarray::stack(Axis(0), &ivar_views)
                        .map_err(|e| PyValueError::new_err(format!("Stack error: {}", e)))?,
                )
            } else {
                None
            };

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
    pub fn read_single<'py>(
        &self,
        py: Python<'py>,
        index: i64,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Option<Bound<'py, PyArray4<bool>>>,
        Option<Bound<'py, PyArray4<f32>>>,
    )> {
        let addr = self
            .get_example_addr(index as usize)
            .ok_or_else(|| MsirError::IndexOutOfBounds(index))?;

        // zarr 读取为纯 Rust I/O，释放 GIL；insert_axis 属于零成本视图操作，顺带一起放入闭包。
        let (flux_4d, mask_4d, ivar_4d) = py.detach(|| -> PyResult<(
            Array4<f32>,
            Option<Array4<bool>>,
            Option<Array4<f32>>,
        )> {
            let (flux, mask, ivar) = self.read_single_example(&addr.0, addr.1)?;
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
    pub fn get_addr(&self, index: i64) -> Option<(String, usize)> {
        if index < 0 {
            return None;
        }
        self.get_example_addr(index as usize)
    }

    /// 查询索引范围内的所有 interval
    pub fn query_intervals<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: usize,
    ) -> Bound<'py, PyArray1<i64>> {
        let results: Vec<i64> = self
            .index
            .find(start, end)
            .map(|iv| iv.start as i64)
            .collect();
        results.into_pyarray(py)
    }

    /// 收集样本 ID
    ///
    /// 训练模型时，会将数据分配到不同的进程（GPU）进行训练，每个进程需要读取不同的数据子集。
    /// 本方法根据进程 rank 和进程总数，计算出每个进程需要读取的样本 ID 范围。
    ///
    /// Args:
    ///     rank: 进程 rank
    ///     world_size: 进程总数
    ///
    /// Returns:
    ///     list[range]: 每个元素为一个 Python range 对象
    pub fn collect_example_ids<'py>(
        &self,
        py: Python<'py>,
        rank: usize,
        world_size: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let builtins = py.import("builtins")?;
        let range_type = builtins.getattr("range")?;
        let list = PyList::empty(py);
        for iv in self.index.iter() {
            let start = iv.start;
            let end = iv.stop;
            // Python: range(start, end)[rank::world_size] == range(start + rank, end, world_size)
            let shard_start = start + rank;
            if shard_start < end {
                let r = range_type.call1((shard_start, end, world_size))?;
                list.append(r)?;
            }
        }
        Ok(list)
    }

    /// 估算分片后的批次数量
    ///
    /// Args:
    ///     batch_size: 批次大小
    ///     world_size: 总进程数
    ///     num_workers: 每个进程的 worker 数
    ///
    /// Returns:
    ///     int: 最小分片的批次数量
    pub fn estimate_sharded_batches(&self, batch_size: usize, world_size: usize, num_workers: usize) -> usize {
        let total_slots = world_size * num_workers;
        let mut sizes = vec![0usize; total_slots];

        for rank in 0..world_size {
            for worker in 0..num_workers {
                for iv in self.index.iter() {
                    let start = iv.start;
                    let end = iv.stop;
                    // range(start, end)[rank::world_size] 的长度
                    let shard_start = start + rank;
                    if shard_start >= end {
                        continue;
                    }
                    let shard_len = (end - shard_start + world_size - 1) / world_size;
                    // 再按 worker 分片: range(0, shard_len)[worker::num_workers]
                    let worker_start = worker;
                    if worker_start >= shard_len {
                        continue;
                    }
                    let worker_len = (shard_len - worker_start + num_workers - 1) / num_workers;
                    sizes[rank * num_workers + worker] += worker_len;
                }
            }
        }

        let min_size = sizes.iter().copied().min().unwrap_or(0);
        min_size / batch_size
    }
}

impl AstroImageReader {
    /// 构建索引
    fn build_index(
        zarr_root_path: &str,
        max_chunk_size: usize,
    ) -> PyResult<(Lapper<usize, IntervalData>, Vec<String>, usize, usize)> {
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
                .last()
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

        Ok((lapper, subsets, total_samples, num_channels))
    }

    /// 获取样本地址
    fn get_example_addr(&self, example_id: usize) -> Option<(String, usize)> {
        let mut results = self.index.find(example_id, example_id + 1);

        results.next().map(|interval| {
            let local_idx = example_id - interval.start + interval.val.offset;
            (interval.val.subset_path.clone(), local_idx)
        })
    }

    /// 读取单个样本
    fn read_single_example(
        &self,
        subset_path: &str,
        local_idx: usize,
    ) -> PyResult<(Array3<f32>, Option<Array3<bool>>, Option<Array3<f32>>)> {
        let store = build_store(&self.zarr_root_path)?;

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
}

/// Python 模块定义。
///
/// 使用声明式模块语法（`#[pymodule] mod ...`），以便通过 `#[pymodule_init]`
/// 在模块导入阶段完成 tokio runtime 的一次性初始化。
#[pymodule]
mod msir {
    use pyo3::exceptions::PyImportError;
    use pyo3::prelude::*;
    use tokio::runtime::Runtime;

    // 将 `AstroImageReader` 暴露到 Python 模块命名空间。
    #[pymodule_export]
    use super::AstroImageReader;

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
        let rt = Runtime::new().map_err(|e| {
            PyImportError::new_err(format!(
                "failed to create tokio runtime for msir: {}",
                e
            ))
        })?;
        super::RUNTIME.set(rt).map_err(|_| {
            PyImportError::new_err("msir tokio runtime is already initialized")
        })?;
        Ok(())
    }
}
