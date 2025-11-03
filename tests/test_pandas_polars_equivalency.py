from portfolio import portfolio_from_file

def test_pandas_vs_polars_equivalence():
    json_path = "portfolio_structure-1.json"
    csv_path = "market_data-1.csv"

    p_pandas = portfolio_from_file(json_path, csv_path, use_polars=False, threaded=False)
    p_polars = portfolio_from_file(json_path, csv_path, use_polars=True, threaded=False)

    dict_pd = p_pandas.to_dict()
    dict_pl = p_polars.to_dict()

    assert abs(dict_pd["total_value"] - dict_pl["total_value"]) < 1e-8
    assert abs(dict_pd["aggregate_volatility"] - dict_pl["aggregate_volatility"]) < 1e-8
    assert abs(dict_pd["max_drawdown"] - dict_pl["max_drawdown"]) < 1e-8

    for pd_pos, pl_pos in zip(dict_pd["positions"], dict_pl["positions"]):
        assert pd_pos["symbol"] == pl_pos["symbol"]
        assert abs(pd_pos["value"] - pl_pos["value"]) < 1e-8
        assert abs(pd_pos["volatility"] - pl_pos["volatility"]) < 1e-8
        assert abs(pd_pos["drawdown"] - pl_pos["drawdown"]) < 1e-8
