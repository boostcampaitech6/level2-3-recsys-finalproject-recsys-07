from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import math
import numpy as np
import pandas as pd
import pickle
import requests
import torch
import json
import pymysql
import re

with open("../config/api_key.json", "r") as f:
    conf = json.load(f)
api_key = conf.get("api_key")

with open("../config/DB_config.json", "r") as f:
    DATABASE_CONFIG = json.load(f)
DATABASE_CONFIG["charset"] = "utf8mb4"
DATABASE_CONFIG["cursorclass"] = pymysql.cursors.DictCursor


def fetch_table_data(table_name):

    try:
        connection = pymysql.connect(**DATABASE_CONFIG)

        with connection.cursor() as cursor:
            query = f"SELECT * FROM {table_name}"
            if table_name == "app_info":
                query = """SELECT appid, name, is_adult, is_free, 
on_windows, on_mac, on_linux, English, Korean, multi_player,
PvP, Co_op, MMO,Action, Adventure, Indie,
RPG, Strategy, Simulation,Casual, Sports, Racing,
Violent, Gore FROM app_info;"""
            cursor.execute(query)
            result = cursor.fetchall()

            return pd.DataFrame(result)

    except Exception as e:
        print(f"An error occurred while fetching data from {table_name}: {e}")
        return pd.DataFrame()  # 에러가 발생했다면, 빈 DataFrame 반환

    finally:
        # 연결 종료
        if connection:
            connection.close()


def labeling(time):
    if time == 0:
        return 0
    else:
        base = 10000 * 60
        return math.log(time, base)
        # 2시간 = 1.0, 4시간 = 2.0, ... , 1000시간 = 10.0, 10000시간 = 13.3


def get_user_games(user_id):
    if user_id is None:
        return -1
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
        return -2  # 에러가 발생한 경우


def extract_steam64id_from_url(profile_url):
    # profile에 steam64ID가 있을경우
    match = re.search(r"steamcommunity\.com/profiles/([0-9]+)", profile_url)
    if match:
        return match.group(1)

    # 사용자 정의 URL의 경우 Steam API를 통해 steam64ID 조회
    match = re.search(r"steamcommunity\.com/id/(\w+)", profile_url)
    if match:
        vanity_url = match.group(1)
        response = requests.get(
            f"https://api.steampowered.com/ISteamUser/ResolveVanityURL/v0001/",
            params={"key": api_key, "vanityurl": vanity_url},
        )
        data = response.json()
        if data["response"]["success"] == 1:
            return data["response"]["steamid"]

    return None


@asynccontextmanager
async def service_initialize(app: FastAPI):
    """
    추론을 위한 모델을 로드하고, db에 연결한다.
    """
    # load model
    with open("model/B.pickle", "rb") as f:
        B = pickle.load(f)
    with open("model/app_stat.pickle", "rb") as f:
        df = pickle.load(f)
        col = df.index
    app.state.B = B
    app.state.df = df
    app.state.col = col
    # sql 연결
    app.state.app_info_df = fetch_table_data("app_info")
    app.state.app_info_df["appid"] = app.state.app_info_df["appid"].map(int)
    print("got app info")
    app.state.user_info_df = fetch_table_data("user_info")
    app.state.user_hash = {
        v: idx for idx, v in app.state.user_info_df["steamid"].items()
    }
    print("got user info")
    # app.state.interaction_df = fetch_table_data("user_game_interaction")
    print("got interaction info")
    yield


app = FastAPI(lifespan=service_initialize)

app.mount("/static", StaticFiles(directory="fe"), name="static")

templates = Jinja2Templates(directory="fe")


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict")
async def predict(request: Request, user_urls: str = Query(...)):
    """
    Input: 복수의 steam user 들의 profil url을 comma(,) 를 delimeter로 하여 하나의 str로 입력받습니다.
    Output: Status에 성공 여부 를, Response에 appid, likelihood를 return 합니다.
    """
    B = request.app.state.B
    z_df = request.app.state.df
    col = request.app.state.col
    hash_col = {int(k): True for k in col}
    urls = list(user_urls.split(","))
    user_ids = [extract_steam64id_from_url(url) for url in urls]
    with open("resource/new_id_list.txt", "a") as f:
        for id in user_ids:
            if id not in app.state.user_hash:
                f.write(id + "\n")
    user_libraries = [get_user_games(id) for id in user_ids]

    users_error = [0, 0]
    for i, library in enumerate(user_libraries):
        if library in [-1, -2]:
            users_error[i] = library
        if library == []:
            users_error[i] = -3
    if users_error != [0, 0]:
        return {"errorcode": users_error}
    # input validation
    # get user information from db
    # get user infromation from steam api
    inference_pivot = pd.DataFrame(index=range(len(urls)), columns=col)
    for i, library in enumerate(user_libraries):
        for game in library:
            appid = game["appid"]
            # print(appid)
            if appid not in hash_col:
                continue
            time = labeling(game["playtime_forever"])
            z_score = (
                game["playtime_forever"] - z_df.loc[str(appid), "mean"]
            ) / z_df.loc[str(appid), "std"]
            inference_pivot.loc[i, str(appid)] = max(time + z_score, 0)
    # print(hash_col)
    inference_pivot = inference_pivot.fillna(0)
    X = torch.tensor(inference_pivot.values).to(dtype=torch.float).to("cuda")
    # inference by model
    S = X @ B
    # print(X)
    # print(X.unique())
    # print(S.unique())
    # print(B)
    # post process
    S[0] -= torch.min(S[0])
    S[-1] -= torch.min(S[-1])
    S[0] /= torch.max(S[0])
    S[-1] /= torch.max(S[-1])

    S_ = S[0] * S[-1]

    idx2item = {i: v for i, v in enumerate(col)}

    _, idx = torch.topk(S_, 1000)

    predict_data = [
        (idx2item[i], S[0][i].item(), S[-1][i].item()) for i in idx.reshape(-1).tolist()
    ]

    df = pd.DataFrame(predict_data, columns=["appid", "p1likelihood", "p2likelihood"])
    # print(df[df["p1likelihood"].isna()])
    for i, library in enumerate(user_libraries):
        appid2idx = {appid: idx for idx, appid in df["appid"].items()}
        df[f"p{i+1}own"] = 0
        for game in library:
            try:
                df.iloc[appid2idx[str(game["appid"])], 3 + i] = 1
            except:
                pass
    # print(app.state.app_info_df[app.state.app_info_df["appid"].isna()])
    # print(df[df["appid"] == df["appid"].max()])
    df["appid"] = df["appid"].astype("int")
    df = df.merge(app.state.app_info_df, on="appid")
    df["total_preference"] = df["p1likelihood"] + df["p2likelihood"]

    df["preference_ratio1"] = 100 * df["p1likelihood"] / df["total_preference"]
    df["preference_ratio1"] = df["preference_ratio1"].astype("int")

    df["preference_ratio2"] = 100 * df["p2likelihood"] / df["total_preference"]
    df["preference_ratio2"] = df["preference_ratio2"].astype("int")
    predict_with_metadata = json.loads(df.to_json(orient="records"))

    return predict_with_metadata


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
