import pandas as pd
import math
import torch
import pickle

import warnings

warnings.filterwarnings("ignore")


def labeling(time):
    if time == 0:
        return 0.5
    elif time < 120:
        return 1.0
    else:
        hour = time / 60
        return math.log(hour, 2)
        # 2시간 = 1.0, 4시간 = 2.0, ... , 1000시간 = 10.0, 10000시간 = 13.3


def train(lambda_: int = 100, log_labeling: bool = True) -> None:
    lambda_ = lambda_

    df = pd.read_csv("DB_interaction.csv")

    if log_labeling:
        df["playtime_forever"] = df["playtime_forever"].apply(labeling)
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
    with open("Column.pickle", "wb") as f:
        pickle.dump(pivot.columns, f)


if __name__ == "__main__":
    train(lambda_=100)
