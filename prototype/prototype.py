import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import threading
from EASE import *
import json


def get_user_games(user_id):
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


def load_image(url, game_id, images_dict):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    images_dict[game_id] = img


def show_recommendations():
    images = {}
    result_arr = st.session_state.result[:10]  # 최대 10개의 추천 결과만 사용

    image_urls = {
        f"https://cdn.akamai.steamstatic.com//steam//apps//{game_id}//header.jpg?t=1666290860": game_id
        for game_id in result_arr
    }

    threads = []
    for url, game_id in image_urls.items():  # 이미지 로딩을 위한 스레드 생성
        thread = threading.Thread(target=load_image, args=(url, game_id, images))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if images:  # 이미지가 있을 경우에만 열 생성
        cols = st.columns(3)  # 3열로 이미지 표시
        col_index = 0
        for game_id, img in images.items():
            if col_index == len(cols):  # 현재 행의 열이 모두 찼으면, 새 행 생성
                cols = st.columns(3)
                col_index = 0
            col = cols[col_index]
            col_index += 1
            if col.image(img, use_column_width=True):
                url = f"https://store.steampowered.com/app/{game_id}"
                col.markdown(
                    f'<a href="{url}" target="_blank">Link to Game</a>',
                    unsafe_allow_html=True,
                )


# main 함수에서는 페이지네이션 로직을 제거하고, 단순히 show_recommendations를 호출합니다.
def main():
    if "show_recommendations" not in st.session_state:
        st.session_state.show_recommendations = False

    st.title("게임 추천 시스템")

    if st.session_state.show_recommendations:
        show_recommendations()
        # "처음으로" 버튼 추가
        if st.button("처음으로"):
            st.session_state.show_recommendations = False
            st.experimental_rerun()

    else:
        st.session_state.user_code = st.text_input("player1의 코드를 입력하세요.")
        st.session_state.friend_code = st.text_input("plater2의 코드를 입력하세요.")
        if st.button("추천 받기"):
            if st.session_state.user_code and st.session_state.friend_code:
                st.session_state.user_games = get_user_games(st.session_state.user_code)
                st.session_state.friend_games = get_user_games(
                    st.session_state.friend_code
                )

                if (
                    st.session_state.user_games == -1
                    and st.session_state.friend_games == -1
                ):
                    st.error(
                        "player1의 게임 정보를 가져올 수 없습니다. Steam ID를 확인해주세요."
                    )
                    st.error(
                        "player2의 게임 정보를 가져올 수 없습니다. Steam ID를 확인해주세요."
                    )
                elif st.session_state.user_games == -1:
                    st.error(
                        "player1의 게임 정보를 가져올 수 없습니다. Steam ID를 확인해주세요."
                    )
                elif st.session_state.friend_games == -1:
                    st.error(
                        "player2의 게임 정보를 가져올 수 없습니다. Steam ID를 확인해주세요."
                    )
                else:
                    st.session_state.result = ease(
                        st.session_state.user_games, st.session_state.friend_games
                    )
                    st.session_state.show_recommendations = True
                    st.experimental_rerun()
            else:
                st.error("모든 필드를 입력해주세요.")


if __name__ == "__main__":
    with open("../config/api_key.json", "r") as f:
        conf = json.load(f)
    api_key = conf.get("api_key")
    # main 호출
    main()
