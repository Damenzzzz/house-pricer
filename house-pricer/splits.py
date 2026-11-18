import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 5
GROUP_COL = "duplicate_group_id"
STRAT_COL = "district"
SEED = 0


def _groups(df: pd.DataFrame) -> np.ndarray:
    if GROUP_COL in df and df[GROUP_COL].notna().any():
        g = df[GROUP_COL].astype("string")
        return g.fillna(pd.Series(df.index.astype(str), index=df.index)).to_numpy()
    return df.index.astype(str).to_numpy()


def _strata(df: pd.DataFrame) -> np.ndarray:
    city = df["city"].astype("string").fillna("?")
    district = df[STRAT_COL].astype("string").fillna("не указан") if STRAT_COL in df else "?"
    s = (city + " / " + district).to_numpy()
    counts = pd.Series(s).value_counts()
    rare = set(counts[counts < N_SPLITS].index)
    return np.array(["прочее" if x in rare else x for x in s])


def add_folds(df: pd.DataFrame, n_splits=N_SPLITS, seed=SEED) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for k, (_, test_idx) in enumerate(cv.split(df, _strata(df), _groups(df))):
        df.loc[test_idx, "fold"] = k
    assert (df.fold >= 0).all()
    return df


def train_test_split_grouped(df: pd.DataFrame, test_fold=0, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_folds(df, **kw)
    return df[df.fold != test_fold].copy(), df[df.fold == test_fold].copy()


def check_split(df: pd.DataFrame) -> pd.DataFrame:
    groups = pd.Series(_groups(df), index=df.index)
    leaked = groups.groupby(groups).apply(lambda g: df.loc[g.index, "fold"].nunique() > 1).sum()
    print(f"групп, размазанных по фолдам: {leaked} (должно быть 0)")
    print(f"размеры фолдов: {df.fold.value_counts().sort_index().to_dict()}")
    share = pd.crosstab(df[STRAT_COL], df.fold, normalize="columns").round(3)
    print("\nдоля района внутри каждого фолда (столбцы должны быть похожи):")
    print(share.to_string())
    return share
