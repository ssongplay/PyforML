import numpy as np
import pandas as pd


def get_rating_matrix(filename, dtype=np.float32):
    df = pd.read_csv(filename)
    return df.groupby(["source", "target"])["rating"].sum().unstack().fillna(0)


print(get_rating_matrix("movie_rating.csv"))


def get_frequent_matrix(filename, dtype=np.float32):
    pass
