import torch
import threading
import queue


class ThreadedScheduler():
    """
    Usage:

    with ThreadedScheduler(scheduler) as data_iterator:
        for data in data_iterator:
            # do something with data

    """

    def __init__(self, scheduler, queue_size: int = 4):
        self.data_queue = queue.Queue(maxsize=queue_size)
        self.scheduler_thread = threading.Thread(target=self.loadData)
        self.scheduler = scheduler

    def loadData(self):
        for data_tuple in self.scheduler:
            data_copy = []
            for each in data_tuple:
                if isinstance(each, torch.Tensor):
                    data_copy.append(each.clone())
                else:
                    data_copy.append([item.clone() for item in each])

            self.data_queue.put(data_copy)

    def __len__(self):
        return len(self.scheduler)

    def __enter__(self):
        self.scheduler_thread.start()
        return self

    def __iter__(self):
        for _ in range(len(self.scheduler)):
            yield self.data_queue.get()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # error handle
        self.scheduler_thread.join()