import os
import json
from typing import Union

import matplotlib.pyplot as plt
from datetime import datetime


class ResultsLogger:
    def __init__(self, filepath=None, truncate_fields=None, max_length=500, top_k=None, mode=None):
        """
        Initialize the logger. If filepath is not given, generate one using timestamp, top_k, and mode.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = ["results"]

        if mode:
            filename_parts.append(mode)
        if top_k:
            filename_parts.append(f"top{top_k}")
        filename_parts.append(timestamp)

        auto_filename = "_".join(filename_parts) + ".jsonl"
        self.filepath = filepath or os.path.join("results", auto_filename)
        self.max_length = max_length
        self.truncate_fields = truncate_fields or {"query", "answer", "context"}

        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def _truncate(self, value):
        if isinstance(value, str) and len(value) > self.max_length:
            return value[:self.max_length] + "...[truncated]"
        return value

    def log(self, record: Union[dict, str]):
        if isinstance(record, str):
            safe_record = {"message": record}
        else:
            safe_record = {}
            for k, v in record.items():
                # Handle truncation fields first
                if k in self.truncate_fields and isinstance(v, str):
                    safe_record[k] = self._truncate(v)
                    continue
                # Handle numpy arrays
                try:
                    import numpy as np
                    if isinstance(v, np.ndarray):
                        safe_record[k] = v.tolist()
                        continue
                except ImportError:
                    pass
                # Handle common complex types
                if hasattr(v, "to_dict") and callable(getattr(v, "to_dict")):
                    safe_record[k] = v.to_dict()
                elif hasattr(v, "text"):
                    safe_record[k] = v.text
                elif isinstance(v, (list, tuple)):
                    # Try to convert each item to string (for lists of complex objects)
                    safe_record[k] = [str(item) for item in v]
                elif isinstance(v, (int, float, bool)) or v is None:
                    safe_record[k] = v
                else:
                    safe_record[k] = str(v)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")

    def load_results(self):
        """Load all logged results from the output file."""
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    def summarize_scores(self):
        """Prints basic statistics (count, average, min, max) for similarity scores."""
        results = self.load_results()
        scores = [r["score"] for r in results if "score" in r and isinstance(r["score"], (int, float))]
        if not scores:
            print("No scores to summarize.")
            return

        print(f"\n📊 Results Summary:")
        print(f"Total entries with score: {len(scores)}")
        print(f"Average score: {sum(scores) / len(scores):.4f}")
        print(f"Min score: {min(scores):.4f}")
        print(f"Max score: {max(scores):.4f}")


# --- Histogram plotting function ---
def plot_score_distribution(filepath="results/outputs.jsonl"):
    """Plot a histogram of similarity scores from the results file."""
    if not os.path.exists(filepath):
        print(f"No results file found at {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        scores = [
            json.loads(line.strip()).get("score")
            for line in f if line.strip()
        ]

    scores = [s for s in scores if isinstance(s, (int, float))]

    if not scores:
        print("No valid scores found to plot.")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=10, color="skyblue", edgecolor="black")
    plt.title("Similarity Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
