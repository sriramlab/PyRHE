import multiprocessing
import signal
import sys
import time
from tqdm import tqdm
import torch

class MultiprocessingHandler:
    def __init__(self, target, work_ranges, device):
        self.target = target
        self.work_ranges = work_ranges
        self.device = device
        self.processes = []

    def _signal_handler(self, sig, frame):
        for p in self.processes:
            try:
                if p.is_alive() and p._popen is not None:
                    p.terminate()
            except AssertionError:
                pass
        sys.exit(1)

    def start_processes(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        if self.device == torch.device("cuda"):
            multiprocessing.set_start_method('spawn', force=True)
        for worker_num, (start_j, end_j) in enumerate(self.work_ranges):
            p = multiprocessing.Process(target=self.target, args=(worker_num, start_j, end_j))
            self.processes.append(p)
            p.start()

    def join_processes(self):
        try:
            for p in tqdm(self.processes, desc="Preprocessing jackknife subsamples..."):
                p.join()
                if p.exitcode != 0:
                    raise Exception("A worker process failed.")
        except Exception as e:
            print(f"Error: {e}. Terminating all processes...")
            for p in self.processes:
                if p.is_alive():
                    p.terminate()
            sys.exit(1)