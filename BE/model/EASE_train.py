import math

import torch
import torch.nn as nn
import pandas as pd
import pickle


class EASE(nn.Module):
    def __init__(self, n_items, col):
        super(EASE, self).__init__()
        self.register_buffer("B", torch.randn(n_items, n_items))
        self.register_buffer("mask", torch.ones(n_items, n_items) - torch.eye(n_items))

    def get_closed_form(self, lambda_: int, X):
        G = X.T @ X
        G += torch.eye(G.shape[0]).to("cuda") * lambda_
        P = G.inverse()
        B = P / (-1 * P.diag())
        B = B * self.mask
        self.B.data = B

    def forward(self, input):
        return input @ self.B


def labeling(time):
    if time == 0:
        return 0.5
    elif time < 120:
        return 1.0
    else:
        hour = time / 60
        return math.log(hour, 2)
        # 2시간 = 1.0, 4시간 = 2.0, ... , 1000시간 = 10.0, 10000시간 = 13.3


def load_data(path):
    df = pd.read_csv(path)

    df["playtime_forever"] = df["playtime_forever"].apply(labeling)
    pivot = df.pivot(
        index="steamid", columns="appid", values="playtime_forever"
    ).fillna(0)
    X = torch.tensor(pivot.values).to(dtype=torch.float).to("cuda")
    return X, pivot.columns


def main():
    data, col = load_data("DB_interaction.csv")
    model = EASE(data.shape[1], col).to("cuda")
    model.get_closed_form(100, data)
    model_scripted = torch.jit.script(model)  # TorchScript 형식으로 내보내기
    model_scripted.save("model_scripted.pt")  # 저장하기
    with open("Column.pickle", "wb") as f:
        pickle.dump(col, f)


if __name__ == "__main__":
    main()
