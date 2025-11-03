import json
import numpy as np
import pandas as pd
import polars as pl
from concurrent.futures import ThreadPoolExecutor
from data_loader import load_price_data
from metrics import compute_volatility, compute_max_drawdown


class Position:
    def __init__(self, symbol, quantity, price, data=None):
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.data = data
        self.value = quantity * price
        self.volatility = None
        self.drawdown = None

    def compute_metrics(self, use_polars=False):
        if self.data is None or len(self.data) == 0:
            return
        if use_polars:
            returns = self.data.select(pl.col("price").pct_change().drop_nulls())["price"].to_numpy()
        else:
            returns = self.data["price"].pct_change().dropna().values
        self.volatility = compute_volatility(returns)
        self.drawdown = compute_max_drawdown(returns)


class Portfolio:
    def __init__(self, name, owner=None, positions=None, sub_portfolios=None):
        self.name = name
        self.owner = owner
        self.positions = positions or []
        self.sub_portfolios = sub_portfolios or []
        self.total_value = None
        self.aggregate_volatility = None
        self.max_drawdown = None

    def compute_aggregate_metrics(self):
        if not self.positions:
            self.total_value = 0
            self.aggregate_volatility = 0
            self.max_drawdown = 0
            return

        total_value = sum(p.value for p in self.positions)
        self.total_value = total_value
        weights = np.array([p.value / total_value for p in self.positions])
        vols = np.array([p.volatility for p in self.positions])
        dds = np.array([p.drawdown for p in self.positions])
        self.aggregate_volatility = np.nansum(weights * vols)
        self.max_drawdown = np.nansum(weights * dds)

    def to_dict(self):
        d = {
            "name": self.name,
            "total_value": self.total_value,
            "aggregate_volatility": self.aggregate_volatility,
            "max_drawdown": self.max_drawdown,
            "positions": [
                {
                    "symbol": p.symbol,
                    "value": p.value,
                    "volatility": p.volatility,
                    "drawdown": p.drawdown,
                }
                for p in self.positions
            ],
        }
        if self.sub_portfolios:
            d["sub_portfolios"] = [sp.to_dict() for sp in self.sub_portfolios]
        return d

    def build_sequential(self, json_data, price_data, use_polars=False):
        self.positions = [
            create_position(pos, price_data, use_polars) for pos in json_data.get("positions", [])
        ]
        self.sub_portfolios = [
            Portfolio(sub["name"]).build_sequential(sub, price_data, use_polars)
            for sub in json_data.get("sub_portfolios", [])
        ]
        self.compute_aggregate_metrics()
        return self

    def build_threaded(self, json_data, price_data, use_polars=False, max_workers=4):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self.positions = list(
                executor.map(
                    lambda pos: create_position(pos, price_data, use_polars),
                    json_data.get("positions", []),
                )
            )
        self.sub_portfolios = [
            Portfolio(sub["name"]).build_threaded(sub, price_data, use_polars, max_workers)
            for sub in json_data.get("sub_portfolios", [])
        ]
        self.compute_aggregate_metrics()
        return self


def create_position(pos, price_data, use_polars):
    symbol = pos["symbol"]
    if use_polars:
        df = price_data.filter(pl.col("symbol") == symbol)
    else:
        df = price_data[price_data["symbol"] == symbol].copy()

    if use_polars and "price" not in df.columns:
        df = df.rename({df.columns[-1]: "price"})
    elif not use_polars and "price" not in df.columns:
        df.rename(columns={df.columns[-1]: "price"}, inplace=True)

    p = Position(symbol, pos["quantity"], pos["price"], df)
    p.compute_metrics(use_polars=use_polars)
    return p


def portfolio_from_file(json_path, csv_path, use_polars=False, threaded=False, max_workers=4):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    price_data = load_price_data(csv_path, use_polars=use_polars)
    portfolio = Portfolio(json_data["name"], json_data.get("owner"))
    if threaded:
        portfolio.build_threaded(json_data, price_data, use_polars, max_workers)
    else:
        portfolio.build_sequential(json_data, price_data, use_polars)
    return portfolio
