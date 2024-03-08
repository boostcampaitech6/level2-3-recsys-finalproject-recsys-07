import pandas as pd
import numpy as np
import math
import torch
from tqdm import tqdm

from datetime import datetime

from typing import List, Dict

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
    """
    score: 유저의 상위 k개 item에 대한 playtime_forever 값이 담긴 list. 값이 없을 경우 0.
    """
    score = score[:k]
    discounts = np.log2(np.arange(2, len(score) + 2))
    dcg = np.sum(score / discounts)
    return dcg


def ndcg_at_k(score, k=10):
    """
    score: 유저의 상위 k개 item에 대한 playtime_forever 값이 담긴 list. 값이 없을 경우 0.
    """
    actual_dcg = dcg_at_k(score, k)
    sorted_score = np.sort(score)[::-1]
    best_dcg = dcg_at_k(sorted_score, k)
    ndcg = actual_dcg / best_dcg
    return ndcg


def ease(user1_arr: List[Dict], user2_arr: List[Dict]) -> List[int]:
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
    inference_pivot = pd.DataFrame(columns=pivot.columns)
    for game in user1_arr:
        appid = game["appid"]
        if not appid in pivot.columns:
            continue
        time = labeling(game["playtime_forever"])
        inference_pivot.loc[0, appid] = time

    for game in user2_arr:
        appid = game["appid"]
        if not appid in pivot.columns:
            continue
        time = labeling(game["playtime_forever"])
        inference_pivot.loc[1, appid] = time

    inference_pivot = inference_pivot.fillna(0)
    X = torch.tensor(inference_pivot.values).to(dtype=torch.float).to("cuda")
    S = X @ B
    S /= torch.max(S)

    S = S[0] + S[-1]

    print("Recommend")
    idx2item = {i: v for i, v in enumerate(pivot.columns)}

    value, idx = torch.topk(S, 10)

    """user_arr = pivot.index.repeat(10)
    rec_df = pd.DataFrame(
        zip(user_arr, idx.reshape(-1).tolist(), value.reshape(-1).tolist()),
        columns=["user", "item", "preference"],
    )
    
    rec_df.item = rec_df.item.map(idx2item)"""

    end_time = datetime.now()
    print(f"full time: {end_time - start_time}")  # 약 8.5s
    print(f"inference time: {end_time - start_time2}")  # 약 0.1s

    """rec_df.to_csv("ease_output.csv", index=False)"""

    """sum_ndcg = 0
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
    print(f"ndcg@{k} = {sum_ndcg/cnt}")"""

    return [idx2item[i] for i in idx.reshape(-1).tolist()]


if __name__ == "__main__":
    user1_arr = [
        {"appid": 400, "playtime_forever": 243},
        {"appid": 8930, "playtime_forever": 27426},
        {"appid": 620, "playtime_forever": 540},
        {"appid": 206440, "playtime_forever": 10},
        {"appid": 108600, "playtime_forever": 7076},
        {"appid": 294100, "playtime_forever": 10814},
        {"appid": 286160, "playtime_forever": 2893},
        {"appid": 323190, "playtime_forever": 683},
        {"appid": 361280, "playtime_forever": 109},
        {"appid": 261570, "playtime_forever": 0},
        {"appid": 387290, "playtime_forever": 591},
        {"appid": 413150, "playtime_forever": 3216},
        {"appid": 413410, "playtime_forever": 1071},
        {"appid": 413420, "playtime_forever": 1400},
        {"appid": 420530, "playtime_forever": 300},
        {"appid": 431960, "playtime_forever": 804},
        {"appid": 457140, "playtime_forever": 35993},
        {"appid": 477160, "playtime_forever": 40},
        {"appid": 289070, "playtime_forever": 16361},
        {"appid": 527230, "playtime_forever": 560},
        {"appid": 558990, "playtime_forever": 1482},
        {"appid": 568220, "playtime_forever": 126},
        {"appid": 646570, "playtime_forever": 15282},
        {"appid": 648800, "playtime_forever": 350},
        {"appid": 424840, "playtime_forever": 758},
        {"appid": 677120, "playtime_forever": 70},
        {"appid": 736260, "playtime_forever": 609},
        {"appid": 1049590, "playtime_forever": 321},
        {"appid": 1057090, "playtime_forever": 0},
        {"appid": 1061090, "playtime_forever": 186},
        {"appid": 285900, "playtime_forever": 64},
        {"appid": 1115690, "playtime_forever": 342},
        {"appid": 1116580, "playtime_forever": 91},
        {"appid": 1147560, "playtime_forever": 653},
        {"appid": 1145360, "playtime_2weeks": 34, "playtime_forever": 915},
        {"appid": 1210030, "playtime_forever": 698},
        {"appid": 1222140, "playtime_forever": 771},
        {"appid": 1256670, "playtime_forever": 152},
        {"appid": 1301390, "playtime_forever": 1523},
        {"appid": 1372810, "playtime_forever": 3493},
        {"appid": 1404850, "playtime_forever": 312},
        {"appid": 1435790, "playtime_forever": 46},
        {"appid": 1562700, "playtime_forever": 849},
        {"appid": 1568590, "playtime_forever": 488},
        {"appid": 1623730, "playtime_2weeks": 375, "playtime_forever": 731},
        {"appid": 1714040, "playtime_forever": 463},
        {"appid": 1811990, "playtime_2weeks": 924, "playtime_forever": 4001},
        {"appid": 1868140, "playtime_forever": 4105},
        {"appid": 1942280, "playtime_forever": 2942},
        {"appid": 1948280, "playtime_forever": 535},
        {"appid": 916440, "playtime_forever": 0},
        {"appid": 2296990, "playtime_forever": 0},
        {"appid": 2379780, "playtime_2weeks": 544, "playtime_forever": 544},
    ]
    user2_arr = [
        {"appid": 10, "playtime_forever": 15070},
        {"appid": 20, "playtime_forever": 34},
        {"appid": 30, "playtime_forever": 39},
        {"appid": 40, "playtime_forever": 0},
        {"appid": 50, "playtime_forever": 0},
        {"appid": 60, "playtime_forever": 3},
        {"appid": 70, "playtime_forever": 13},
        {"appid": 130, "playtime_forever": 0},
        {"appid": 80, "playtime_forever": 1},
        {"appid": 100, "playtime_forever": 0},
        {"appid": 220, "playtime_forever": 622},
        {"appid": 240, "playtime_forever": 1621},
        {"appid": 320, "playtime_forever": 11},
        {"appid": 340, "playtime_forever": 0},
        {"appid": 2500, "playtime_forever": 0},
        {"appid": 11200, "playtime_forever": 0},
        {"appid": 12520, "playtime_forever": 0},
        {"appid": 37800, "playtime_forever": 0},
        {"appid": 34820, "playtime_forever": 0},
        {"appid": 550, "playtime_forever": 2811},
        {"appid": 24960, "playtime_forever": 70},
        {"appid": 38700, "playtime_forever": 0},
        {"appid": 20500, "playtime_forever": 44},
        {"appid": 43110, "playtime_forever": 0},
        {"appid": 667720, "playtime_forever": 0},
        {"appid": 10680, "playtime_forever": 0},
        {"appid": 50130, "playtime_forever": 0},
        {"appid": 1030830, "playtime_forever": 0},
        {"appid": 57900, "playtime_forever": 0},
        {"appid": 730, "playtime_forever": 12728},
        {"appid": 109600, "playtime_forever": 0},
        {"appid": 218230, "playtime_forever": 0},
        {"appid": 1083500, "playtime_forever": 0},
        {"appid": 238960, "playtime_forever": 0},
        {"appid": 252950, "playtime_forever": 772},
        {"appid": 39150, "playtime_forever": 37},
        {"appid": 273110, "playtime_forever": 0},
        {"appid": 323370, "playtime_forever": 0},
        {"appid": 353190, "playtime_forever": 97},
        {"appid": 232090, "playtime_forever": 174},
        {"appid": 418460, "playtime_forever": 158},
        {"appid": 391220, "playtime_forever": 212},
        {"appid": 446850, "playtime_forever": 0},
        {"appid": 218620, "playtime_forever": 0},
        {"appid": 739630, "playtime_forever": 296},
        {"appid": 1085660, "playtime_forever": 0},
        {"appid": 203160, "playtime_forever": 0},
        {"appid": 2073850, "playtime_forever": 10},
        {"appid": 1938090, "playtime_forever": 19},
        {"appid": 2357570, "playtime_forever": 8},
    ]
    result = ease(user1_arr, user2_arr)
    print(result)
