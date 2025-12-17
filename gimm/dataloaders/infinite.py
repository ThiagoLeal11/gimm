from typing import Generator
import threading
import queue

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

    def __iter__(self) -> Generator[BATCH, None, None]:
        loader = self.dataloader
        if self.is_infinite:
            loader = self._infinite(loader)

        if self.should_preload:
            loader = self._preload(loader)
        else:
            loader = map(self._to_device, loader)

        for batch in loader:
            yield batch

    def _infinite(self, dataloader: DL) -> Generator[BATCH, None, None]:
        while True:
            for (imgs, labels) in dataloader:
                # Discard the last batch if it is not a full batch
                if imgs.size(0) == self.batch_size:
                    yield imgs, labels
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
