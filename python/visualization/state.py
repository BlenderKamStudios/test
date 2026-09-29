"""State shared by the live displays (live.py, dashboard.py): per-receiver
processing pipelines, the baseline manager and the change history."""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from preprocessing.baseline import BaselineManager
from preprocessing.pipeline import ReceiverPipeline


class LiveState:
    """Everything the figures need; fed record by record."""

    def __init__(self, cfg, source_label, sim=None):
        self.cfg = cfg
        self.source = source_label
        self.sim = sim
        hs = cfg["visualization"]["history_seconds"]
        self.pipelines = {r["id"]: ReceiverPipeline(cfg["preprocessing"], hs) for r in cfg["receivers"]}
        self.baseline = BaselineManager(cfg["baseline"], self.pipelines.keys())
        self._change = {i: deque(maxlen=20000) for i in self.pipelines}
        self.now = time.time()
        self.stats_text = ""
        self.truth = None

    def feed(self, rec):
        p = self.pipelines.get(rec.receiver_id)
        if p is None:
            return
        f = p.process(rec)
        if f is None:
            return
        self.now = f.t                          # data time (replay/simulation run at any speed)
        self.baseline.feed(rec.receiver_id, f)
        rc = self.baseline.rx[rec.receiver_id]
        if self.baseline.state == "ready" and np.isfinite(rc.ratio):
            self._change[rec.receiver_id].append((f.t, rc.ratio))

    def change_history(self, rx_id):
        d = self._change[rx_id]
        horizon = self.cfg["visualization"]["history_seconds"] * 3
        while d and self.now - d[0][0] > horizon:
            d.popleft()
        if not d:
            return np.array([]), np.array([])
        a = np.array(d)
        return a[:, 0], a[:, 1]

    @property
    def clock(self):
        return time.strftime("%H:%M:%S", time.localtime(self.now))
