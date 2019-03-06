import torch.utils.data
from torch.utils.data.sampler import BatchSampler

from mtcnn.config import cfg


class IterationBasedBatchSampler(BatchSampler):
    def __init__(self, batch_sampler, max_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.max_iterations = max_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.max_iterations:
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.max_iterations:
                    break
                yield batch

    def __len__(self):
        return self.max_iterations


def build_batch_sampler(dataset, batch_size, shuffle=False):
    if shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    sampler = BatchSampler(sampler, batch_size, drop_last=False)
    batch_sampler = IterationBasedBatchSampler(sampler, cfg.TRAIN.MAX_ITERS)
    return batch_sampler
