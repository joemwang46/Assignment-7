from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import pandas as pd
import polars as pl


def threading_pd(metric, df: pd.DataFrame, symbols: list, max_workers=4):
    transformed = []

    for symbol in symbols:
        symbol_data = df.loc[df['symbol'] == symbol, 'price'].reset_index(drop=True)
        transformed.append(symbol_data)
    
    df_new = pd.concat(transformed, axis=1)
    df_new.columns = symbols

    window = 20
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
        executor.submit(metric, df_new[symbol], window): symbol
        for symbol in df_new.columns
    }

    for future in as_completed(futures):
        symbol = futures[future]
        result = future.result()
        results.append(result)

    final_df = pd.concat(results, axis =1, ignore_index=True)


def threading_pl(metric, df: pl.DataFrame, symbols: list, max_workers=4):
    transformed = {}

    for symbol in symbols:
        symbol_data = (
            df.filter(pl.col("symbol") == symbol)
              .select(pl.col("price"))
              .to_series()
        )
        transformed[symbol] = symbol_data

    df_new = pl.DataFrame(transformed)

    window = 20
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(metric, df_new[symbol], window): symbol
            for symbol in df_new.columns
        }

        for future in as_completed(futures):
            symbol = futures[future]
            result = future.result()
            results.append(result.rename(symbol))

    final_df = pl.DataFrame().hstack(results)

    return final_df

def multiprocessing_pd(metric, df: pd.DataFrame, symbols: list, max_workers=4):
    transformed = []

    for symbol in symbols:
        symbol_data = df.loc[df['symbol'] == symbol, 'price'].reset_index(drop=True)
        transformed.append(symbol_data)

    df_new = pd.concat(transformed, axis=1)
    df_new.columns = symbols

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(metric, df_new[symbol]): symbol
            for symbol in df_new.columns
        }

        for future in as_completed(futures):
            symbol = futures[future]
            result = future.result()
            results.append(result)

    final_df = pd.concat(results, axis=1, ignore_index=True)
    return final_df


def multiprocessing_pl(metric, df: pl.DataFrame, symbols: list, max_workers=4):
    transformed = {}
    for symbol in symbols:
        symbol_data = df.filter(pl.col("symbol") == symbol)["price"]
        transformed[symbol] = symbol_data

    df_new = pl.DataFrame(transformed)

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(metric, df_new[symbol]): symbol
            for symbol in df_new.columns
        }

        for future in as_completed(futures):
            symbol = futures[future]
            result = future.result()

            if isinstance(result, pl.DataFrame):
                result = result.rename({col: f"{symbol}_{col}" for col in result.columns})
            elif isinstance(result, pl.Series):
                result = result.rename(symbol)

            results.append(result)

    final_df = pl.concat(results, how="horizontal")
    return final_df


