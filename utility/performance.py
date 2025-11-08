# utility/performance.py
import time
import psutil
import torch
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List
import json
from utility.logger import logger


class PerformanceMonitor:
    """Real-time performance tracking for RAG operations"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_session = {}

    def start_operation(self, operation_name: str):
        """Start timing an operation"""
        self.current_session[operation_name] = {
            'start_time': time.time(),
            'start_memory': self.get_memory_usage(),
            'start_gpu_memory': self.get_gpu_memory()
        }

    def end_operation(self, operation_name: str) -> Dict:
        """End timing and record metrics"""
        if operation_name not in self.current_session:
            logger.warning(f"Operation {operation_name} was never started")
            return {}

        session = self.current_session[operation_name]
        duration = time.time() - session['start_time']
        memory_delta = self.get_memory_usage() - session['start_memory']
        gpu_memory_delta = self.get_gpu_memory() - session['start_gpu_memory']

        metrics = {
            'operation': operation_name,
            'duration': duration,
            'memory_delta_mb': memory_delta,
            'gpu_memory_delta_mb': gpu_memory_delta,
            'timestamp': time.time()
        }

        self.metrics[operation_name].append(metrics)
        del self.current_session[operation_name]

        logger.info(
            f"Performance {operation_name}: {duration:.2f}s, Memory: {memory_delta:+.1f}MB, GPU: {gpu_memory_delta:+.1f}MB")
        return metrics

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return psutil.Process().memory_info().rss / 1024 / 1024

    def get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.backends.mps.is_available():
            # MPS doesn't have detailed memory reporting
            return 0.0
        elif torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def get_summary(self) -> Dict:
        """Get performance summary"""
        summary = {}
        for operation, measurements in self.metrics.items():
            if measurements:
                durations = [m['duration'] for m in measurements]
                summary[operation] = {
                    'count': len(measurements),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_duration': sum(durations)
                }
        return summary

    def save_metrics(self, filepath: str = "performance_metrics.json"):
        """Save metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
        logger.info(f"Performance metrics saved to {filepath}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


@contextmanager
def monitor_performance(operation_name: str):
    """Context manager for monitoring performance"""
    performance_monitor.start_operation(operation_name)
    try:
        yield
    finally:
        performance_monitor.end_operation(operation_name)


def performance_report():
    """Print a performance report"""
    summary = performance_monitor.get_summary()

    print("\nPERFORMANCE REPORT")
    print("=" * 50)

    for operation, stats in summary.items():
        print(f"\nOperation: {operation}")
        print(f"   Count: {stats['count']}")
        print(f"   Average: {stats['avg_duration']:.2f}s")
        print(f"   Min: {stats['min_duration']:.2f}s")
        print(f"   Max: {stats['max_duration']:.2f}s")
        print(f"   Total: {stats['total_duration']:.2f}s")

    return summary


# Decorator for automatic performance monitoring
def track_performance(operation_name: str = None):
    """Decorator to automatically track function performance"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            with monitor_performance(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
