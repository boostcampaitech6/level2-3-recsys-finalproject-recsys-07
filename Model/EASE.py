import pandas as pd
import numpy as np
import math
import torch
from tqdm import tqdm

from datetime import datetime

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


def dcg_at_k(score, k=10):
    score = score[:k]
    discounts = np.log2(np.arange(2, len(score) + 2))
    dcg = np.sum(score / discounts)
    return dcg


def ndcg_at_k(score, k=10):
    actual_dcg = dcg_at_k(score, k)
    sorted_score = np.sort(score)[::-1]
    best_dcg = dcg_at_k(sorted_score, k)
    ndcg = actual_dcg / best_dcg
    return ndcg


def main():
    start_time = datetime.now()

    lambda_ = 100

    df = pd.read_csv("user_game_interaction.csv")

    df["playtime_forever"] = df["playtime_forever"].apply(labeling)
    pivot = df.pivot(
        index="steam_id", columns="appid", values="playtime_forever"
    ).fillna(0)

    print("EASE")
    X = torch.tensor(pivot.values).to(dtype=torch.float).to("cuda")
    G = X.T @ X

    G += torch.eye(G.shape[0]).to("cuda") * lambda_

    P = G.inverse()

    B = P / (-1 * P.diag())
    for i in range(len(B)):
        B[i][i] = 0

    start_time2 = datetime.now()
    S = X @ B
    S /= torch.max(S)

    print("Recommend")
    value, idx = torch.topk(S, 10)
    user_arr = pivot.index.repeat(10)
    rec_df = pd.DataFrame(
        zip(user_arr, idx.reshape(-1).tolist(), value.reshape(-1).tolist()),
        columns=["user", "item", "preference"],
    )

    idx2item = {i: v for i, v in enumerate(pivot.columns)}
    rec_df.item = rec_df.item.map(idx2item)

    end_time = datetime.now()
    print(f"full time: {end_time - start_time}")  # 약 8.5s
    print(f"inference time: {end_time - start_time2}")  # 약 0.1s

    rec_df.to_csv("ease_output.csv", index=False)

    sum_ndcg = 0
    cnt = 0
    k = 10
    for user, group in tqdm(rec_df.groupby("user").item):
        cnt += 1
        score = []
        for item in group.values:
            try:
                playtime = df[df["steam_id"] == user][
                    df["appid"] == item
                ].playtime_forever.values[0]
            except:
                playtime = 0
            score.append(playtime)
        ndcg = ndcg_at_k(score, k=k)
        sum_ndcg += ndcg
    print(f"ndcg@{k} = {sum_ndcg/cnt}")


if __name__ == "__main__":
    main()
