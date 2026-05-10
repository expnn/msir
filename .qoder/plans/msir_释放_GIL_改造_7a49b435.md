# msir 释放 GIL 改造

## 背景

当前 [lib.rs](file:///Users/cyc/Sources/MultiSurveyImageEncoder/src/msir/src/lib.rs) 中的 `#[pymethods] impl AstroImageReader` 暴露的函数在执行远程 OSS / 本地磁盘 I/O 与 rayon 并行计算时，始终持有 GIL，限制了 Python 侧多线程并发（如 PyTorch DataLoader worker）。使用 PyO3 新 API `Python::detach` 将纯 Rust 重活与 GIL 解耦。

## 改造范围

仅改动 `#[pymethods] impl AstroImageReader` 中下列三个入口，`impl AstroImageReader` 里 [build_index](file:///Users/cyc/Sources/MultiSurveyImageEncoder/src/msir/src/lib.rs#L427) / [read_single_example](file:///Users/cyc/Sources/MultiSurveyImageEncoder/src/msir/src/lib.rs#L526) / [get_example_addr](file:///Users/cyc/Sources/MultiSurveyImageEncoder/src/msir/src/lib.rs#L516) 等内部函数不动。

不改动项：getters（total_samples、num_channels、crop_size、disable_mask、disable_ivar、subsets）、[get_addr](file:///Users/cyc/Sources/MultiSurveyImageEncoder/src/msir/src/lib.rs#L405)、[query_intervals](file:///Users/cyc/Sources/MultiSurveyImageEncoder/src/msir/src/lib.rs#L410)。

## Task 1: 改造 [new](file:///Users/cyc/Sources/MultiSurveyImageEncoder/src/msir/src/lib.rs#L226)（`__init__`）

- 在签名中加入 `py: Python<'_>`（PyO3 `#[new]` 支持 Python token 参数）。
- 将 `Self::build_index(&zarr_root_path, max_chunk_size)?` 包在 `py.detach(|| Self::build_index(&zarr_root_path, max_chunk_size))?` 中。
- `build_index` 返回 `(Lapper<...>, Vec<String>, usize, usize)`，均 `Send`，满足 `detach` 约束。
- `PyResult` 中的 `PyErr` 本身 `Send`，可跨闭包边界向外传播。

## Task 2: 改造 [read_single](file:///Users/cyc/Sources/MultiSurveyImageEncoder/src/msir/src/lib.rs#L377)

- 保持当前签名（已有 `py: Python<'py>`）。
- GIL 下：保留 `get_example_addr` 调用与 `IndexOutOfBounds` 错误构造（均为内存操作，无需释放）。
- `py.detach(|| -> PyResult<(Array4<f32>, Option<Array4<bool>>, Option<Array4<f32>>)> { ... })` 内：
  - 调用 `self.read_single_example(&addr.0, addr.1)?` 得到三个 `Array3`；
  - 对它们分别执行 `insert_axis(Axis(0))` 得到 `Array4`；
  - 返回三元组。
- 回到 GIL 下：对每个 `Array4` 调用 `into_pyarray(py)`，返回现有 tuple 结构。

## Task 3: 改造 [read_batch](file:///Users/cyc/Sources/MultiSurveyImageEncoder/src/msir/src/lib.rs#L297)

- 保持当前签名。
- GIL 下：`let idx_slice: &[i64] = indices.as_slice()?;`。该切片指向 numpy 底层 C 缓冲区，`PyReadonlyArray1` 在 Rust 侧维护借用跟踪；`&[i64]` 满足 `Send + Sync + Ungil`，可直接跨 `detach` 边界使用，无需 `to_vec()` 拷贝。
- **线程安全假设（策略 A + 文档约定）**：`PyReadonlyArray1` 的借用跟踪只约束 Rust 侧，不阻止其他 Python 线程通过 Python API 修改 `indices`。释放 GIL 后若有并发线程改写 `indices` 会产生数据竞争。采纳社区常规做法：由调用方保证 `read_batch` 执行期间不修改传入的 `indices` 数组，并在该方法的 Python doc 中新增一行中文说明，例如：`注意: 调用期间内部会释放 GIL，调用方必须保证在本次调用返回前不从其他线程修改 indices 数组，否则行为未定义。`
- 同时在 [read_single](file:///Users/cyc/Sources/MultiSurveyImageEncoder/src/msir/src/lib.rs#L377) 的 doc 中简短补充：`注意: 调用期间内部会释放 GIL。` （该方法的 `index` 参数为标量 `i64`，按值拷贝，无并发修改风险，但仍需提示行为变化。）
- `py.detach(|| -> PyResult<(Array4<f32>, Option<Array4<bool>>, Option<Array4<f32>>)> { ... })` 内：
  1. 基于 `idx_slice` 生成 `addrs`;
  2. 通过 rayon `into_par_iter` 调度 `read_single_example`；
  3. 收集 `fluxes` / `masks` / `ivars`；对空集合与不一致场景继续使用当前逻辑（返回 `PyValueError` 或 `None`）；
  4. 执行 `ndarray::stack(Axis(0), &views)` 生成 `flux_4d` / `mask_4d` / `ivar_4d`；
  5. 将三者作为三元组返回。
- 回到 GIL 下：对 `Array4` 调用 `into_pyarray(py)`，组装为最终 tuple。

## 注意事项

- PyO3 版本：使用 `Python::detach`（新 API，替代 `Python::allow_threads`），无需额外 feature。构建时若编译器提示 `detach` 不存在，回退到 `allow_threads` 并在提交说明中标注。
- `read_batch` 的 numpy `indices` 参数：释放 GIL 后不做防御性拷贝，依赖文档约定调用方不得并发修改该数组（见 Task 3）。
- `AstroImageReader` 字段（`String`、`Arc<Lapper<usize, IntervalData>>`、`Vec<String>`、基本类型、`bool`）均为 `Send + Sync`，`&self` 可安全穿越 `detach`。
- `detach` 闭包内不得捕获/返回任何 `Bound<'py, ...>`、`Py<T>`、`PyAny` 等 GIL-bound 类型；闭包只返回 `ndarray` 与 `PyErr`。
- 不修改 [build_store](file:///Users/cyc/Sources/MultiSurveyImageEncoder/src/msir/src/lib.rs#L90)、`TokioBlockOn`、`RUNTIME` 初始化等既有路径。tokio runtime 的 `block_on` 现在发生在非 GIL 线程，进一步降低 GIL + block_on 交互的潜在风险。

## 验收

1. `cargo build` / `cargo check` 通过，无新增告警（允许既有告警保持原状）。
2. 代码风格与现有文件一致：中文 doc 注释保持、`PyResult` 错误映射不变、返回的 PyArray 类型签名不变。
3. 无 `unsafe` 新增。
4. 不新增 `unwrap`, `expect`, `abort`, `panic!` 调用。