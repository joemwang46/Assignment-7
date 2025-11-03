import numpy as np
from portfolio import portfolio_from_file

def test_portfolio_aggregation_totals():
    json_path = "portfolio_structure-1.json"
    csv_path = "market_data-1.csv"

    p = portfolio_from_file(json_path, csv_path, use_polars=False, threaded=False)
    d = p.to_dict()

    values = np.array([pos["value"] for pos in d["positions"]])
    vols = np.array([pos["volatility"] for pos in d["positions"]])
    dds = np.array([pos["drawdown"] for pos in d["positions"]])

    total_value = np.sum(values)
    weights = values / total_value
    expected_agg_vol = np.nansum(weights * vols)
    expected_agg_dd = np.nansum(weights * dds)

    assert np.isclose(d["total_value"], total_value, atol=1e-8)
    assert np.isclose(d["aggregate_volatility"], expected_agg_vol, atol=1e-8)
    assert np.isclose(d["max_drawdown"], expected_agg_dd, atol=1e-8)
