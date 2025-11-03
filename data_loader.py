from datetime import datetime
import pandas as pd
import polars as pl

def load_price_data(filename, use_polars=False):
    if use_polars:
        df = pl.read_csv(filename)
    else:
        df = pd.read_csv(filename)
    return df