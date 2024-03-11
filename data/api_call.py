import argparse
import requests
import time
from datetime import datetime
from tqdm import tqdm
import pickle
from datetime import datetime


# 스팀 BaseID 부터 i를 증가시킨 아이디를 반환 (i가 1이상일때, 제대로 작동)
def baseID(i):
    baseID = 76561197960265728
    baseID += i
    str_ID = str(baseID)
    return str_ID


# 스팀 유저의 게임 라이브러리를 가져오는 함수
# name, appid(game id로 추정), playtime_forever(총 게임 플레이 시간)을 list of dictionary로 가지고 옴
def get_user_game_library(steam_api_key, steam_user_id):
    url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": steam_api_key,
        "steamid": steam_user_id,
        "format": "json",
        "include_appinfo": True,
        "include_played_free_games": True,
    }
    response = requests.get(url, params=params)
    # 조회가 성공한 경우 status_code == 200
    if response.status_code == 200:
        data = response.json()
        games = data.get("response", {}).get("games", [])
        # 유저 아이디가 비어 있는 경우 비어있는 list ([])를 반환
        return [
            (
                steam_user_id,
                game.get("name", "nan"),
                game.get("appid", "nan"),
                game.get("playtime_forever", 0),
                game.get("playtime_2weeks", 0),
                game.get("img_logo_url", "nan"),
            )
            for game in games
        ]
    # 조회가 실패한 경우 알림을 띄우고 -1 반환
    elif response.status_code == 429:
        print("End of daily API call allowance.")
        current_time = datetime.now()
        print(f"gug:{current_time.hour}:{current_time.minute}")
        return -1
    else:
        print(response.status_code, ": Failed to retrieve data.")
        current_time = datetime.now()
        print(f"gug:{current_time.hour}:{current_time.minute}")
        return 1


# 스팀 유저의 정보를 가져오는 함수
def get_user_summary(steam_api_key, steam_user_id):
    url = "http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
    params = {"key": steam_api_key, "steamids": steam_user_id, "format": "json"}
    response = requests.get(url, params=params)
    # 조회가 성공한 경우 status_code == 200
    if response.status_code == 200:
        data = response.json()
        players = data.get("response", {}).get("players", [])
        # 유저 아이디가 비어 있는 경우 비어있는 list ([])를 반환
        return [
            (steam_user_id, player.get("personaname", "nan"), datetime.now())
            for player in players
        ]
    # 조회가 실패한 경우 알림을 띄우고 -1 반환
    elif response.status_code == 429:
        print("End of daily API call allowance.")
        current_time = datetime.now()
        print(f"gus:{current_time.hour}:{current_time.minute}")
        return -1
    else:
        print(response.status_code, ": Failed to retrieve data.")
        current_time = datetime.now()
        print(f"gus:{current_time.hour}:{current_time.minute}")
        return 1


# 스팀 게임의 추가 정보를 가져오는 함수
def get_app_add_info(app_id):
    url = "https://store.steampowered.com/api/appdetails"
    params = {"appids": app_id}
    response = requests.get(url, params=params)
    # 조회가 성공한 경우 status_code == 200
    if response.status_code == 200:
        data = response.json()
        items = data.get(app_id, {}).get("data", [])
        # 유저 아이디가 비어 있는 경우 비어있는 list ([])를 반환
        return items
    #         return [
    #             {
    #                 #사용할 FEATURE 목록 DICT 형식으로 추가

    #             }
    #             for item in items
    #         ]
    # 조회가 실패한 경우 알림을 띄우고 -1 반환
    else:
        print("Failed to retrieve data.")
        return -1


def main(args):
    api_version = args.version
    start_i = int(args.start_i)
    range_dict = {
        1: range(start_i, 337507060, 800),
        2: range(start_i, 675005294, 800),
        3: range(start_i, 1012503528, 800),
        4: range(start_i, 1350001762, 800),
        5: range(start_i, 1687500000, 800),
    }
    range_list = range_dict[api_version]
    steam_api_key = args.api_key

    # user list를 담을 dictionary 생성
    user_game_list = []
    user_profile = []
    print("start")
    for i in tqdm(range_list):
        steam_id = baseID(i)
        temp_list = get_user_game_library(steam_api_key, steam_id)

        if temp_list == -1:  # 조회가 실패한 경우 반복문 continue 하는 것으로 수정
            print("library_i:", i)
            break
        elif temp_list == 1:
            continue
        else:  # 조회가 성공한 경우
            time.sleep(1)  # API 호출을 위한 속도 조절
            if temp_list:
                user_game_list.extend(temp_list)

                # 게임 정보가 있는 유저에 한해서 유저 정보 가져오기
                temp_list = get_user_summary(steam_api_key, steam_id)
                if temp_list == -1:
                    print("profile_i:", i)
                    break
                elif temp_list == 1:
                    continue
                else:
                    time.sleep(1)  # API 호출을 위한 속도 조절
                    if temp_list:
                        user_profile.extend(temp_list)

        current_time = datetime.now()
        # 오전 08시 50분에 종료 (UTC 23:50)
        if current_time.hour == 8 and current_time.minute == 50:
            print(f"{current_time.hour}:{current_time.minute}")
            print("Stop the execution:", i)
            break
    # print(user_game_list)

    with open(f"user_games_{api_version}.pickle", "wb") as fw:
        pickle.dump(user_game_list, fw)
    with open(f"user_profile_{api_version}.pickle", "wb") as fw:
        pickle.dump(user_profile, fw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to choose execution version.")
    parser.add_argument(
        "--version", action="store", dest="version", type=int, help="choose version"
    )
    parser.add_argument(
        "--api_key", action="store", dest="api_key", type=str, help="API_KEY"
    )
    parser.add_argument("--start_i", action="store", dest="start_i", type=str, help="")
    args = parser.parse_args()

    # main 함수 호출
    main(args)
