"""Lazy model loading and VRAM hygiene for a 4GB GPU.

Models are created on first use, never at import time. Every inference runs
under a single lock (so concurrent requests cannot stack activations on the
GPU) and empties the CUDA allocator cache when it finishes.
"""
import contextlib
import hashlib
import threading
from collections import OrderedDict

import numpy as np
import torch

GPU_LOCK = threading.RLock()


def release_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def inference():
    with GPU_LOCK, torch.inference_mode():
        try:
            yield
        finally:
            release_vram()


class LazyModel:
    def __init__(self, name, loader):
        self.name = name
        self._loader = loader
        self._obj = None
        self._lock = threading.Lock()

    def get(self):
        if self._obj is None:
            with self._lock:
                if self._obj is None:
                    print(f"Loading {self.name}...")
                    self._obj = self._loader()
                    release_vram()
        return self._obj


class LRUCache:
    """Tiny per-image result cache, so repeated clicks / re-renders skip inference."""

    def __init__(self, size=4):
        self.size = size
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)


def image_key(image_np):
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(image_np).data)
    h.update(str(image_np.shape).encode())
    return h.hexdigest()
