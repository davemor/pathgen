import pandas as pd


def simple_random(class_df: pd.DataFrame, sum_totals: int) -> pd.DataFrame:
    class_sample = class_df.sample(n=sum_totals, axis=0, replace=False)
    return class_sample
