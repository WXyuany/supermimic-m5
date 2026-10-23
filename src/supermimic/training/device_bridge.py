"""Move sampled replay batches to a separate learner device without changing sampling."""
from dataclasses import fields

import torch


class LearnerReplay:
    def __init__(self, source, device):
        self.source = source
        self.device = torch.device(device)

    @property
    def size(self):
        return self.source.size

    def _move(self, batch):
        return type(batch)(**{f.name: getattr(batch, f.name).to(self.device) for f in fields(batch)})

    def sample_windows(self, n, k):
        return self._move(self.source.sample_windows(n, k))

    def sample_starts(self, n, hist):
        return self._move(self.source.sample_starts(n, hist))


def synchronize(device):
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "mps":
        torch.mps.synchronize()
