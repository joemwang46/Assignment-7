import os
import json
from portfolio import portfolio_from_file
from reporting import (
    compare_ingestion_times,
    compare_rolling_metrics,
    compare_parallel_execution,
    summarize_all
)

def main():
    json_path = "portfolio_structure-1.json"
    csv_path = "market_data-1.csv"

    print("\n==============================")
    print("  PORTFOLIO PERFORMANCE TESTS")
    print("==============================\n")

    # --- Step 1: Ingestion Comparison (Pandas vs Polars)
    print("[1/4] Comparing data ingestion performance...")
    ingestion_df = compare_ingestion_times(csv_path)
    print(ingestion_df)
    print()

    # --- Step 2: Rolling Metric Computation (Volatility + Drawdown)
    print("[2/4] Comparing rolling metric computation...")
    rolling_df = compare_rolling_metrics(csv_path, symbol="AAPL")
    print(rolling_df)
    print()

    # --- Step 3: Portfolio Build — Sequential vs Threaded
    print("[3/4] Comparing sequential vs threaded portfolio execution...")
    parallel_df = compare_parallel_execution(json_path, csv_path)
    print(parallel_df)
    print()

    # --- Step 4: Run All Benchmarks Together
    print("[4/4] Running complete summary...")
    summary = summarize_all(csv_path, json_path)

    print("\n============ SUMMARY ============")
    print("Ingestion:")
    print(summary["ingestion"])
    print("\nRolling Metrics:")
    print(summary["rolling_metrics"])
    print("\nParallel Execution:")
    print(summary["parallel_exec"])

    # --- Step 5: Build example portfolios (for demonstration)
    print("\nBuilding example portfolios...\n")
    portfolio_pd = portfolio_from_file(json_path, csv_path, use_polars=False, threaded=False)
    portfolio_pl = portfolio_from_file(json_path, csv_path, use_polars=True, threaded=True)

    print("Sequential (Pandas) Portfolio:")
    print(json.dumps(portfolio_pd.to_dict(), indent=2))

    print("\nThreaded (Polars) Portfolio:")
    print(json.dumps(portfolio_pl.to_dict(), indent=2))

    print("\n✅ Benchmarking complete.\n")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"  # prevent OpenBLAS thread spam
    main()
