from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import math
import pandas as pd
import pickle
import requests
import torch


def labeling(time):
    if time == 0:
        return 0.5
    elif time < 120:
        return 1.0
    else:
        hour = time / 60
        return math.log(hour, 2)
        # 2시간 = 1.0, 4시간 = 2.0, ... , 1000시간 = 10.0, 10000시간 = 13.3


def get_user_games(user_id):
    api_key = ""
    url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={api_key}&steamid={user_id}&format=json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # 네트워크 에러나 4xx, 5xx 응답을 에러로 처리
        data = response.json()
        if "response" in data and "games" in data["response"]:
            return data["response"]["games"]
        else:
            return []  # API는 성공했으나 게임 목록이 없는 경우
    except requests.exceptions.RequestException as e:
        return -1  # 에러가 발생한 경우


@asynccontextmanager
async def service_initialize(app: FastAPI):
    """
    추론을 위한 모델을 로드하고, db에 연결한다.
    """
    # load model
    model = torch.jit.load("model/model_scripted.pt")
    model.eval()
    with open("model/Column.pickle", "rb") as f:
        model.col = pickle.load(f)
    app.state.model = model
    # sql 연결
    yield


app = FastAPI(lifespan=service_initialize)


@app.get("/predict/{user_urls}")
async def predict(user_urls: str, request: Request):
    """
    Input: 복수의 steam user 들의 profil url을 comma(,) 를 delimeter로 하여 하나의 str로 입력받습니다.
    Output: return 합니다.
    """
    model = request.app.state.model
    col = model.col
    hash_col = {k: True for k in col}
    urls = list(user_urls.split(","))
    user_libraries = [get_user_games(url) for url in urls]

    # input validation
    # get user information from db
    # get user infromation from steam api
    inference_pivot = pd.DataFrame(columns=col)
    for library in user_libraries:
        for game in library:
            appid = game["appid"]
            if appid not in hash_col:
                continue
            time = labeling(game["playtime_forever"])
            inference_pivot.loc[0, appid] = time

    inference_pivot = inference_pivot.fillna(0)
    X = torch.tensor(inference_pivot.values).to(dtype=torch.float).to("cuda")
    # inference by model
    S = model(X)
    # post process
    S[0] -= torch.min(S[0])
    S[-1] -= torch.min(S[-1])
    S[0] /= torch.max(S[0])
    S[-1] /= torch.max(S[-1])

    S_ = S[0] * S[-1]

    idx2item = {i: v for i, v in enumerate(col)}

    _, idx = torch.topk(S_, 12)

    return {
        "dev": "progress",
        "reponse": [
            (idx2item[i], S[0][i].item(), S[-1][i].item())
            for i in idx.reshape(-1).tolist()
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# todo
# input -> url -> get steamid64 -> Error check -> get library and play time -> Error checklr ->
# model
# latency check
