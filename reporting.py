import time
import psutil
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_price_data
from metrics import compute_volatility, compute_max_drawdown
from portfolio import portfolio_from_file


class BenchmarkResult:
    def __init__(self, name, duration, memory_mb, cpu_percent):
        self.name = name
        self.duration = duration
        self.memory_mb = memory_mb
        self.cpu_percent = cpu_percent

    def to_dict(self):
        return {
            "name": self.name,
            "duration_sec": round(self.duration, 4),
            "memory_mb": round(self.memory_mb, 2),
            "cpu_percent": round(self.cpu_percent, 2),
        }


def measure_performance(func, *args, **kwargs):
    process = psutil.Process()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start
    mem_after = process.memory_info().rss / (1024 ** 2)
    cpu_after = process.cpu_percent(interval=None)
    cpu_util = (cpu_after + cpu_before) / 2
    mem_used = max(mem_after - mem_before, 0)
    return result, duration, mem_used, cpu_util


def compare_ingestion_times(csv_path):
    results = []
    for mode in [("Pandas", False), ("Polars", True)]:
        _, duration, mem, cpu = measure_performance(load_price_data, csv_path, mode[1])
        results.append(BenchmarkResult(f"{mode[0]} Load", duration, mem, cpu))
    df = pd.DataFrame([r.to_dict() for r in results])
    df.plot(kind="bar", x="name", y="duration_sec", title="Ingestion Time Comparison", legend=False)
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.show()
    return df


def compare_rolling_metrics(csv_path, symbol="AAPL"):
    pandas_df = load_price_data(csv_path, use_polars=False)
    polars_df = load_price_data(csv_path, use_polars=True)

    pandas_symbol = pandas_df[pandas_df["symbol"] == symbol]
    polars_symbol = polars_df.filter(pl.col("symbol") == symbol)

    def compute_metrics_pandas():
        r = pandas_symbol["price"].pct_change().dropna()
        for _ in range(50):
            compute_volatility(r)
            compute_max_drawdown(r)

    def compute_metrics_polars():
        r = polars_symbol.select(pl.col("price").pct_change().drop_nulls())["price"].to_numpy()
        for _ in range(50):
            compute_volatility(r)
            compute_max_drawdown(r)

    _, t_pd, mem_pd, cpu_pd = measure_performance(compute_metrics_pandas)
    _, t_pl, mem_pl, cpu_pl = measure_performance(compute_metrics_polars)

    data = pd.DataFrame(
        [
            {"library": "Pandas", "time_sec": t_pd, "memory_mb": mem_pd, "cpu_percent": cpu_pd},
            {"library": "Polars", "time_sec": t_pl, "memory_mb": mem_pl, "cpu_percent": cpu_pl},
        ]
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(data["library"], data["time_sec"], color=["#1f77b4", "#ff7f0e"])
    ax.set_title(f"Rolling Metric Computation Time ({symbol})")
    ax.set_ylabel("Seconds")
    plt.tight_layout()
    plt.show()

    return data


def compare_parallel_execution(json_path, csv_path):
    seq_func = lambda: portfolio_from_file(json_path, csv_path, use_polars=False, threaded=False)
    thread_func = lambda: portfolio_from_file(json_path, csv_path, use_polars=False, threaded=True)

    _, t_seq, mem_seq, cpu_seq = measure_performance(seq_func)
    _, t_thread, mem_thread, cpu_thread = measure_performance(thread_func)

    df = pd.DataFrame(
        [
            {"mode": "Sequential", "time_sec": t_seq, "memory_mb": mem_seq, "cpu_percent": cpu_seq},
            {"mode": "Threaded", "time_sec": t_thread, "memory_mb": mem_thread, "cpu_percent": cpu_thread},
        ]
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(df["mode"], df["time_sec"], color=["#2ca02c", "#d62728"])
    ax.set_title("Parallel Execution Speed Comparison")
    ax.set_ylabel("Seconds")
    plt.tight_layout()
    plt.show()

    return df


def summarize_all(csv_path, json_path):
    print("Comparing ingestion performance...")
    ingestion = compare_ingestion_times(csv_path)
    print("\nComparing rolling metric performance...")
    rolling = compare_rolling_metrics(csv_path)
    print("\nComparing parallel vs sequential portfolio build...")
    parallel = compare_parallel_execution(json_path, csv_path)

    summary = {
        "ingestion": ingestion,
        "rolling_metrics": rolling,
        "parallel_exec": parallel,
    }
    return summary
