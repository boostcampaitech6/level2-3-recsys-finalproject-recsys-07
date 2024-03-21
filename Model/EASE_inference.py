import pandas as pd
import numpy as np
import math
import torch
import pickle

from datetime import datetime

from typing import List, Dict, Tuple

import warnings

warnings.filterwarnings("ignore")

from EASE_train import labeling


def ease(user1_arr: List[Dict], user2_arr: List[Dict]) -> List[Tuple[int]]:
    if (not user1_arr) and (not user2_arr):
        return [(-1,)]

    start_time = datetime.now()

    print("Load")
    with open("B.pickle", "rb") as f:
        B = pickle.load(f)
    with open("app_stat.pickle", "rb") as f:
        df = pickle.load(f)
        col = df.index

    start_time2 = datetime.now()
    inference_pivot = pd.DataFrame(columns=col)
    for game in user1_arr:
        appid = game["appid"]
        if not appid in col:
            continue
        time = labeling(game["playtime_forever"])
        z_score = (game["playtime_forever"] - df.loc[appid, "mean"]) / df.loc[
            appid, "std"
        ]
        inference_pivot.loc[0, appid] = max(time + z_score, 0)

    for game in user2_arr:
        appid = game["appid"]
        if not appid in col:
            continue
        time = labeling(game["playtime_forever"])
        z_score = (game["playtime_forever"] - df.loc[appid, "mean"]) / df.loc[
            appid, "std"
        ]
        inference_pivot.loc[1, appid] = max(time + z_score, 0)

    inference_pivot = inference_pivot.fillna(0)
    X = torch.tensor(inference_pivot.values).to(dtype=torch.float).to("cuda")
    S = X @ B
    S[0] -= torch.min(S[0])
    S[-1] -= torch.min(S[-1])
    S[0] /= torch.max(S[0])
    S[-1] /= torch.max(S[-1])

    S_ = S[0] * S[-1]

    print("Recommend")
    idx2item = {i: v for i, v in enumerate(col)}

    _, idx = torch.topk(S_, 12)

    end_time = datetime.now()
    print(f"full time: {end_time - start_time}")
    print(f"inference time: {end_time - start_time2}")

    return [
        (idx2item[i], S[0][i].item(), S[-1][i].item()) for i in idx.reshape(-1).tolist()
    ]


if __name__ == "__main__":
    with open("jh_arr.pickle", "rb") as f:
        user1_arr = pickle.load(f)
    with open("hh_arr.pickle", "rb") as f:
        user2_arr = pickle.load(f)
    result = ease(user1_arr, user2_arr)
    print(result)
