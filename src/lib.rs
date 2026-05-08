use ndarray::{Array3, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray4, PyReadonlyArray1};
use opendal::{Operator, services::S3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rust_lapper::{Interval, Lapper};
use std::env;
use std::future::Future;
use std::sync::{Arc, OnceLock};
use thiserror::Error;
use tokio::runtime::Runtime;
use url::Url;
use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;
use zarrs::storage::ReadableListableStorageTraits;
use zarrs::storage::storage_adapter::async_to_sync::{AsyncToSyncBlockOn, AsyncToSyncStorageAdapter};
use zarrs_opendal::AsyncOpendalStore;

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
        PyValueError::new_err(err.to_string())
    }
}

/// 共享的 tokio runtime，供 AsyncToSyncStorageAdapter 使用。
fn shared_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        Runtime::new().expect("failed to create tokio runtime for msir")
    })
}

/// 将 tokio runtime 适配到 zarrs 的 AsyncToSyncBlockOn trait。
struct TokioBlockOn;

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: Future>(&self, future: F) -> F::Output {
        let _guard = shared_runtime().enter();
        shared_runtime().block_on(future)
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
    let parsed = Url::parse(path).ok();

    match parsed.as_ref().map(|u| u.scheme()) {
        None | Some("") | Some("file") => {
            let local_path = parsed
                .as_ref()
                .map(|u| u.path().to_string())
                .unwrap_or_else(|| path.to_string());
            let store = FilesystemStore::new(&local_path)
                .map_err(|e| PyValueError::new_err(format!("Failed to open filesystem store: {}", e)))?;
            Ok(Arc::new(store))
        }
        Some("oss") => {
            let url = parsed.expect("checked above");

            // 与 Python 端 create_zarr_store 的语义对齐：
            //   - bucket 取 URL path 的第一段；
            //   - endpoint 优先用 URL host（`<scheme>://<host>[:port]`，oss scheme 默认使用 http），
            //     host 为空时回退到 OSS_ENDPOINT 环境变量，再回退到默认 endpoint；
            //   - access_key_id / access_key_secret 先读 URL userinfo，再回退到环境变量。
            let full_path = url.path().trim_start_matches('/').to_string();
            let (bucket, root_in_bucket) = match full_path.split_once('/') {
                Some((b, rest)) if !b.is_empty() => (b.to_string(), rest.to_string()),
                _ if !full_path.is_empty() => (full_path.clone(), String::new()),
                _ => {
                    return Err(PyValueError::new_err(
                        "oss:// URL missing bucket in path (expected oss://[host]/<bucket>/<path>)",
                    ));
                }
            };

            let access_key_id = if !url.username().is_empty() {
                Some(url.username().to_string())
            } else {
                env::var("OSS_ACCESS_KEY_ID").ok()
            };
            let access_key_secret = url
                .password()
                .map(|s| s.to_string())
                .or_else(|| env::var("OSS_ACCESS_KEY_SECRET").ok());

            let endpoint = match url.host_str() {
                Some(host) if !host.is_empty() => {
                    // oss:// 视为未指定具体协议，默认用 http；非 oss scheme 则沿用其 scheme。
                    let scheme = "http";
                    if let Some(port) = url.port() {
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
                .region(&env::var("AWS_REGION").unwrap_or_else(|_| "auto".to_string()));
            if let Some(k) = access_key_id.as_deref() {
                builder = builder.access_key_id(k);
            }
            if let Some(s) = access_key_secret.as_deref() {
                builder = builder.secret_access_key(s);
            }

            let op = Operator::new(builder)
                .map_err(|e| PyValueError::new_err(format!("Failed to build S3 operator: {}", e)))?
                .finish();
            let async_store = Arc::new(AsyncOpendalStore::new(op));
            let sync_store = AsyncToSyncStorageAdapter::new(async_store, TokioBlockOn);
            Ok(Arc::new(sync_store))
        }
        Some(other) => Err(PyValueError::new_err(format!(
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
    /// zarr 数据集根路径（本地路径或 oss:// URL，原样保存）。
    zarr_root_path: String,
    /// interval tree 索引
    index: Arc<Lapper<usize, IntervalData>>,
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
    ///     path: zarr 数据集的根路径，支持本地路径和 `oss://` URL。
    ///     crop_size: 裁剪大小 (默认 96)
    ///     disable_mask: 是否禁用 mask 读取 (默认 False)
    ///     disable_ivar: 是否禁用 ivar 读取 (默认 False)
    ///     max_chunk_size: 最大切片大小，用于将大索引区间切分 (默认 5000)
    #[new]
    #[pyo3(signature = (path, crop_size=96, disable_mask=false, disable_ivar=false, max_chunk_size=5000))]
    pub fn new(
        path: &str,
        crop_size: usize,
        disable_mask: bool,
        disable_ivar: bool,
        max_chunk_size: usize,
    ) -> PyResult<Self> {
        let zarr_root_path = path.trim_end_matches('/').to_string();

        // 构建索引
        let (index, subsets, total_samples, num_channels) =
            Self::build_index(&zarr_root_path, max_chunk_size)?;

        Ok(AstroImageReader {
            zarr_root_path,
            index: Arc::new(index),
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
    pub fn read_batch<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Option<Bound<'py, PyArray4<bool>>>,
        Option<Bound<'py, PyArray4<f32>>>,
    )> {
        let indices = indices.as_slice()?;

        // 预先收集所有需要读取的样本地址
        let addrs: Vec<Option<(String, usize)>> = indices
            .iter()
            .map(|&idx| self.get_example_addr(idx as usize))
            .collect();

        // 并行读取所有样本
        let results: Vec<PyResult<(Array3<f32>, Option<Array3<bool>>, Option<Array3<f32>>)>> = addrs
            .into_par_iter()
            .map(|addr| match addr {
                Some((subset_path, local_idx)) => self.read_single_example(&subset_path, local_idx),
                None => Err(MsirError::IndexOutOfBounds(-1).into()),
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

        // 转换为 PyArray (零拷贝)
        Ok((
            flux_4d.into_pyarray(py),
            mask_4d.map(|m| m.into_pyarray(py)),
            ivar_4d.map(|i| i.into_pyarray(py)),
        ))
    }

    /// 读取单个样本 (返回 numpy 数组)
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

        let (flux, mask, ivar) = self.read_single_example(&addr.0, addr.1)?;

        // 添加 batch 维度
        let flux_4d = flux.insert_axis(Axis(0));
        let mask_4d = mask.map(|m| m.insert_axis(Axis(0)));
        let ivar_4d = ivar.map(|i| i.insert_axis(Axis(0)));

        Ok((
            flux_4d.into_pyarray(py),
            mask_4d.map(|m| m.into_pyarray(py)),
            ivar_4d.map(|i| i.into_pyarray(py)),
        ))
    }

    /// 获取样本的地址 (subset_path, local_idx)
    pub fn get_addr(&self, index: i64) -> Option<(String, usize)> {
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
        let height = shape[2] as usize;
        let width = shape[3] as usize;
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

/// A Python module implemented in Rust.
#[pymodule]
fn msir(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AstroImageReader>()?;
    Ok(())
}
