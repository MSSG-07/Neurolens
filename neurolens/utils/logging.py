"""Small CSV logger for training curves."""
import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def log(self, values):
        exists = self.path.exists()
        with self.path.open("a",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=list(values))
            if not exists: writer.writeheader()
            writer.writerow({k:float(v) if hasattr(v,"item") else v for k,v in values.items()})
