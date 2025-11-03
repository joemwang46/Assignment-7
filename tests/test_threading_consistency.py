from portfolio import portfolio_from_file

def test_threaded_vs_sequential_equivalence(tmp_path):
    json_path = "portfolio_structure-1.json"
    csv_path = "market_data-1.csv"

    p_seq = portfolio_from_file(json_path, csv_path, use_polars=False, threaded=False)
    p_thr = portfolio_from_file(json_path, csv_path, use_polars=False, threaded=True)

    seq_dict = p_seq.to_dict()
    thr_dict = p_thr.to_dict()

    assert abs(seq_dict["total_value"] - thr_dict["total_value"]) < 1e-8
    assert abs(seq_dict["aggregate_volatility"] - thr_dict["aggregate_volatility"]) < 1e-8
    assert abs(seq_dict["max_drawdown"] - thr_dict["max_drawdown"]) < 1e-8

    for seq_pos, thr_pos in zip(seq_dict["positions"], thr_dict["positions"]):
        assert seq_pos["symbol"] == thr_pos["symbol"]
        assert abs(seq_pos["value"] - thr_pos["value"]) < 1e-8
        assert abs(seq_pos["volatility"] - thr_pos["volatility"]) < 1e-8
        assert abs(seq_pos["drawdown"] - thr_pos["drawdown"]) < 1e-8
