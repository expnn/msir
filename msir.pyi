from typing import final

import numpy as np
from _typeshed import Incomplete

@final
class AstroImageReader:
    """
    多巡天图像读取器

    使用 rust-lapper 实现的 interval tree 进行索引，
    使用 zarrs 库高效读取 zarr 格式的天文图像数据。
    """
    def __new__(
        cls,
        /,
        zarr_root_path: str,
        crop_size: int = 96,
        disable_mask: bool = False,
        disable_ivar: bool = False,
        max_chunk_size: int = 5000,
        read_block_size: int = 1,
    ) -> AstroImageReader:
        """
        创建新的 AstroImageReader 实例

        Args:
            zarr_root_path: zarr 数据集的根路径，支持本地路径和 `oss://` URL。
            crop_size: 裁剪大小 (默认 96)
            disable_mask: 是否禁用 mask 读取 (默认 False)
            disable_ivar: 是否禁用 ivar 读取 (默认 False)
            max_chunk_size: 最大切片大小，用于将大索引区间切分 (默认 5000)
        """
    def collect_block_ids(self, /, rank: int, world_size: int) -> list:
        """
        按 block 粒度收集样本 ID。

        每个返回的 ID 代表连续 `block_size` 个样本的起始全局索引。
        子集尾部不足 `block_size` 的样本将被丢弃。

        Args:
            rank: 进程 rank
            world_size: 进程总数
            block_size: 块大小 (= read_block_size)

        Returns:
            list[range]: 每个元素为 block 起始索引的 range
        """
    @property
    def crop_size(self, /) -> int:
        """
        获取裁剪大小
        """
    @property
    def disable_ivar(self, /) -> bool:
        """
        获取是否禁用 ivar
        """
    @property
    def disable_mask(self, /) -> bool:
        """
        获取是否禁用 mask
        """
    def estimate_sharded_batches(self, /, batch_size: int, world_size: int) -> int:
        """
        估算分片后的批次数量（block 粒度版本）

        遍历所有 rank，计算每个 rank 分到的 block 数，取最小值后除以
        每 batch 需要的 block 数 (= batch_size / block_size)，保证所有
        rank 的 batch 数一致，避免 DDP 死锁。

        当 `block_size == 1` 时退化为逐样本的 estimate。

        Args:
            batch_size: 批次大小
            world_size: 总进程数
            block_size: 块大小 (= read_block_size)

        Returns:
            int: 所有 rank 中最小的批次数量
        """
    def get_addr(self, /, index: int) -> tuple[str, int] | None:
        """
        获取样本的地址 (subset_path, local_idx)
        """
    @property
    def num_channels(self, /) -> int:
        """
        获取通道数
        """
    def read_batch(self, /, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None] | None:
        """
        批量读取样本

        当 `read_block_size > 1` 时，indices 中每个值代表一个 block 的起始全局索引，
        实际输出的 batch_size = indices.len() * read_block_size。
        当 `read_block_size == 1` 时，行为与传统逐样本读取等价。

        Args:
            indices: block 起始索引数组 (numpy int64 array)

        Returns:
            tuple: (flux, mask, ivar)
                - flux: shape (N, C, H, W) 的 float32 数组
                - mask: shape (N, C, H, W) 或 (N, 1, H, W) 的 bool 数组，未禁用时返回，否则为 None
                - ivar: shape (N, C, H, W) 的 float32 数组，未禁用时返回，否则为 None

        注意: 调用期间内部会释放 GIL，调用方必须保证在本次调用返回前不从其他
        线程修改 `indices` 数组，否则行为未定义。
        """
    def read_example(self, /, index: np.uintp | int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None] | None:
        """
        读取单个样本 (返回 numpy 数组)

        注意: 调用期间内部会释放 GIL。
        """
    @property
    def subsets(self, /) -> list[str]:
        """
        获取子集列表
        """
    @property
    def total_samples(self, /) -> int:
        """
        获取总样本数
        """

def configure(chunk_concurrent_minimum: int | None = None, codec_concurrent_target: int | None = None) -> None:
    """
    配置 zarrs 内部并发参数。

    Args:
        chunk_concurrent_minimum: chunk 级最小并发数。为 None 时保持不变。
        codec_concurrent_target: codec 编解码总并发上限。为 None 时保持不变。
    """

def __getattr__(name: str) -> Incomplete: ...
