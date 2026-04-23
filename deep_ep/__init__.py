import torch

from .utils import EventOverlap
from .buffer import Buffer

# noinspection PyUnresolvedReferences
from deep_ep_cpp import Config, topk_idx_t


BACKEND_CUDA_NVSHMEM = 'cuda_nvshmem'
BACKEND_SYCL_ISHMEM = 'sycl_ishmem'
SUPPORTED_BACKEND_TERMINOLOGY = (BACKEND_CUDA_NVSHMEM, BACKEND_SYCL_ISHMEM)
