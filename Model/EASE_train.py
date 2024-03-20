import pandas as pd
import math
import torch
import pickle

import warnings

warnings.filterwarnings("ignore")


def labeling(time):
    if time == 0:
        return 0
    else:
        base = 10000 * 60
        return math.log(time, base)
        # 1분 = 0, 2시간 = 0.36, 10시간 = 0.48, 100시간 = 0.65, 10000시간 = 1.00, 40000시간 = 1.10


def train(lambda_: int = 100) -> None:
    lambda_ = lambda_

    print("Load")
    df = pd.read_csv("DB_interaction_0320.csv")

    df["playtime_forever"] = df["playtime_forever"].apply(labeling)
    df["playtime_forever"] += df["z_score"]
    df["playtime_forever"].apply(lambda x: max(x, 0))

    pivot = df.pivot(
        index="steamid", columns="appid", values="playtime_forever"
    ).fillna(0)

    print("EASE")
    X = torch.tensor(pivot.values).to(dtype=torch.float).to("cuda")
    G = X.T @ X

    G += torch.eye(G.shape[0]).to("cuda") * lambda_

    P = G.inverse()

    B = P / (-1 * P.diag())
    for i in range(len(B)):
        B[i][i] = 0

    print("Save")
    with open("B.pickle", "wb") as f:
        pickle.dump(B, f)
    with open("app_stat.pickle", "wb") as f:
        df = pd.DataFrame(df.groupby("appid")[["mean", "std"]].max())
        pickle.dump(df, f)


if __name__ == "__main__":
    train(lambda_=100)
