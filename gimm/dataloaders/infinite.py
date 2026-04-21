from typing import Generator
import threading
import queue
import time
import warnings

import torch

DL = torch.utils.data.DataLoader
BATCH = tuple[torch.Tensor, torch.Tensor]


class InfinitePrefetchLoader:
    def __init__(self, dataloader: DL, device: torch.device = None, *, queue_size: int = 4, infinite: bool = True, preload: bool = True):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.device = device
        self.queue_size = queue_size
        self.epoch = 0

        self.is_infinite = infinite
        self.should_preload = preload
        self._stall_warned = False

    def __iter__(self) -> Generator[BATCH, None, None]:
        yield from self._iter()

    def _iter(self) -> Generator[BATCH, None, None]:
        loader = self.dataloader
        if self.is_infinite:
            loader = self._infinite(loader)

        if self.should_preload:
            preloaded_loader = self._preload(loader)
            loader = self._detect_stalls(preloaded_loader)
        else:
            loader = map(self._to_device, loader)

        for batch in loader:
            yield batch

    def _infinite(self, dataloader: DL) -> Generator[BATCH, None, None]:
        while True:
            batches_seen = 0
            full_batches_seen = 0
            last_batch_size = None
            for (imgs, labels) in dataloader:
                batches_seen += 1
                last_batch_size = imgs.size(0)
                # Discard the last batch if it is not a full batch
                if imgs.size(0) == self.batch_size:
                    full_batches_seen += 1
                    yield imgs, labels

            if batches_seen == 0:
                raise ValueError(
                    "InfinitePrefetchLoader received an empty dataloader and cannot yield batches. "
                    "Check whether the dataset or selected split contains any samples."
                )

            if full_batches_seen == 0:
                raise ValueError(
                    "InfinitePrefetchLoader could not produce a full batch. "
                    f"Expected batch_size={self.batch_size}, but largest observed batch had size {last_batch_size}. "
                    "This usually means the training split is smaller than batch_size (or batch_size * grad_accum_steps). "
                    "Reduce batch_size, reduce grad_accum_steps, or increase the number of training samples in the split."
                )

            self.epoch += 1

    def _to_device(self, batch: BATCH) -> BATCH:
        imgs, labels = batch
        return imgs.to(self.device), labels.to(self.device)

    def _preload(self, dataloader: DL) -> Generator[BATCH, None, None]:
        q_size = self.queue_size
        batch_queue = queue.Queue(maxsize=q_size)

        def producer():
            try:
                for batch_cpu in dataloader:
                    batch_gpu = self._to_device(batch_cpu)
                    batch_queue.put(batch_gpu)
            except Exception as e:
                batch_queue.put(e)
            finally:
                batch_queue.put(None)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while True:
                next_batch = batch_queue.get()
                if next_batch is None:
                    break
                if isinstance(next_batch, Exception):
                    raise next_batch
                yield next_batch
        finally:
            # Ensures the producer thread finishes
            # Even if the consumer is interrupted (e.g., by an external exception or break)
            if producer_thread.is_alive():
                # Attempt to empty the queue to unblock the producer thread if it is stuck in put()
                while not batch_queue.empty():
                    try:
                        batch_queue.get_nowait()
                    except queue.Empty:
                        break
                producer_thread.join(timeout=5.0)

    def _detect_stalls(self, dataloader: Generator[BATCH, None, None]) -> Generator[BATCH, None, None]:
        ema_wait_time = None
        ema_use_time = None
        ema_alpha = 0.2
        stall_ratio = 1.1
        stall_patience = 3
        warmup_batches = 24
        stall_count = 0
        use_time = None
        wait_start = time.perf_counter()
        batches_seen = 0

        for batch in dataloader:
            batches_seen += 1
            wait_time = time.perf_counter() - wait_start

            if batches_seen > warmup_batches:
                ema_wait_time = wait_time if ema_wait_time is None else (1 - ema_alpha) * ema_wait_time + ema_alpha * wait_time
                ema_use_time = use_time if ema_use_time is None else (1 - ema_alpha) * ema_use_time + ema_alpha * use_time

            dataset_loader_stall = (
                batches_seen > warmup_batches
                and
                ema_use_time is not None
                and ema_wait_time > ema_use_time * stall_ratio
            )

            stall_count = stall_count + 1 if dataset_loader_stall else 0
            if stall_count >= stall_patience and not self._stall_warned:
                warnings.warn(
                    f"DataLoader appears to be waiting on CPU-side transforms: "
                    f"Loading average wait={ema_wait_time:.3f}s, average usage time={ema_use_time:.3f}s. "
                    f"Consider increasing the number of workers "
                    f"or enabling `dataset_bake=True` to precompute deterministic transforms."
                )
                self._stall_warned = True

            yielded_at = time.perf_counter()
            yield batch
            resumed_at = time.perf_counter()
            use_time = resumed_at - yielded_at
            wait_start = resumed_at
