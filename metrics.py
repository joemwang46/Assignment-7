import pandas as pd
import polars as pl
import numpy as np

def rolling_ma_pd(series: pd.Series, window=20) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def rolling_ma_pl(series: pl.Series, window=20) -> pl.Series:
    return series.rolling_mean(window_size=window)

def rolling_sd_pd(series: pl.Series, window=20) -> pl.Series:
    return series.rolling(window=window).std()

def rolling_sd_pl(series: pl.Series, window=20) -> pl.Series:
    return series.rolling_std(window_size=window)

def rolling_sharpe_pd(series: pd.Series, window=20) -> pd.Series:
    rolling_mean = rolling_ma_pd(series, window)
    rolling_std = rolling_sd_pd(series, window)
    return rolling_mean / rolling_std

def rolling_sharpe_pl(series: pl.Series, window=20) -> pl.Series:
    rolling_mean = rolling_ma_pl(series, window)
    rolling_std = rolling_sd_pl(series, window)
    return rolling_mean / rolling_std

def compute_volatility(returns, annualize=False, freq=252):
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return np.nan
    vol = np.nanstd(returns, ddof=1)
    if annualize:
        vol *= np.sqrt(freq)
    return vol

def compute_max_drawdown(returns):
    if len(returns) == 0:
        return np.nan
    cumulative = np.cumprod(1 + np.asarray(returns, dtype=float))
    peaks = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / peaks - 1
    return np.nanmin(drawdowns)