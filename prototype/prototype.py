import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import threading
import matplotlib.pyplot as plt
import json

import sys

sys.path.append("/home/jin/level2-3-recsys-finalproject-recsys-07/Model")

from EASE_inference import *


def apply_custom_css():
    st.markdown(
        """
        <style>
            /* 전체 앱 배경색 변경 */
            .stApp {
                background-color: #192A3D;
            }
            /* 버튼 스타일 변경 */
            .stButton>button {
                border: 2px solid #ffffff;
                border-radius: 20px 20px 20px 20px;
                color: #ffffff;
                background-color: #1b2838;
            }
            /* st.title 스타일 변경 */
            h1 {
                color: #ffffff;
                font-family: sans-serif;
            }
            /* st.text_input의 플레이스홀더 글씨 색상 변경 */
            ::placeholder {
                color: #ffffff;
                opacity: 1; /* Firefox */
            }
            .stTextInput>div>div>input::placeholder {
                color: #ffffff; /* Chrome, Safari, Edge, Opera */
            }
            /* st.text_input 위의 라벨 글씨 색상 변경 (직접적인 방법은 제한적) */
            .stTextInput>label {
                color: #ffffff;
            }
            /* st.text_input 스타일 변경 */
            .stTextInput>div>div>input {
                color: #ffffff;
                background-color: #2a475e;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )


def get_game_name(result_list):
    name_list = []
    for result in result_list:
        url = f"https://store.steampowered.com/api/appdetails?appids={result[0]}"
        response = requests.get(url)
        data = response.json()
        if data[f"{result[0]}"]["success"]:
            name_list.append(data[f"{result[0]}"]["data"]["name"])
        else:
            name_list.append("No more service")
    return name_list


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


def load_image(url, result, index, images_list, game_name):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    images_list[index] = (result, game_name, img)  # 게임 ID와 이미지를 튜플로 저장


def show_recommendations():
    num_games = len(st.session_state.result[:12])  # 최대 12개의 추천 결과만 사용
    images_list = [
        (None, None, None)
    ] * num_games  # 이미지와 게임 ID를 저장할 리스트 초기화
    # 선호도 그래프 데이터 준비
    preference_ratios = []

    threads = []
    for index, (result, game_name) in enumerate(
        zip(st.session_state.result[:12], st.session_state.result_name[:12])
    ):
        url = f"https://cdn.akamai.steamstatic.com//steam//apps//{result[0]}//header.jpg?t=1666290860"
        total_preference = result[1] + result[2]  # player1과 player2의 선호도 합계
        preference_ratio = (
            round(result[1] / total_preference * 100),
            round(result[2] / total_preference * 100),
        )
        preference_ratios.append(preference_ratio)

        thread = threading.Thread(
            target=load_image, args=(url, result, index, images_list, game_name)
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # 이미지와 게임 링크 출력
    if images_list:
        cols = st.columns(3)
        col_index = 0
        for index, (result, game_name, img) in enumerate(images_list):
            if result and img:
                if col_index == len(cols):
                    cols = st.columns(3)
                    col_index = 0
                col = cols[col_index]
                col_index += 1
                col.image(img, use_column_width=True)
                url = f"https://store.steampowered.com/app/{result[0]}"
                col.markdown(
                    f'<span style="color: white">{index + 1}.</span>  <a href="{url}" target="_blank">{game_name}</a>',
                    unsafe_allow_html=True,
                )
                player1_ratio, player2_ratio = preference_ratios[index]

                col.markdown(
                    f"""
                        <div style="margin-bottom: 2px;">
                            <span style="color: white">선호도</span><br>
                            <span style="color: #1FC1CC">player1: {int(result[1]*100*0.9)}</span>
                            <span style="color: #EBD834">player2: {int(result[2]*100*0.9)}</span><br>
                            <span style="color: white">취향 반영률</span><br>
                            <span style="color: #1FC1CC">player1 : {player1_ratio}%</span>
                            <span style="color: #EBD834">player2 : {player2_ratio}%</span>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 선호도 그래프 그리기
                fig, ax = plt.subplots(figsize=(4, 1))
                groups = [""]
                # player1_ratio, player2_ratio = preference_ratios[index]
                # Player 1 막대를 음의 방향으로 그립니다.
                ax.barh(groups, [-player1_ratio], color="#1FC1CC", height=1)
                # Player 2 막대를 양의 방향으로 그립니다.
                ax.barh(groups, [player2_ratio], color="#EBD834", height=1)
                ax.set_xlim(
                    -player1_ratio, player2_ratio
                )  # x축의 범위를 설정하여 두 막대가 서로 마주보도록 합니다.
                # plt.axis("off")

                # 배경색 설정
                fig.patch.set_visible(False)  # 외부 배경 제거
                ax.patch.set_visible(False)  # 내부 배경 제거

                # 테두리 제거
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

                # 눈금선 제거 및 눈금 레이블 숨기기
                ax.yaxis.set_ticks_position("none")
                ax.xaxis.set_ticks_position("none")
                ax.set_xticks([])  # x축 눈금 제거
                ax.set_yticks([])  # y축 눈금 제거
                ax.xaxis.set_ticklabels([])  # x축 눈금 레이블 숨기기
                ax.yaxis.set_ticklabels([])  # y축 눈금 레이블 숨기기

                col.pyplot(fig)


# main 함수에서는 페이지네이션 로직을 제거하고, 단순히 show_recommendations를 호출합니다.
def main():
    if "show_recommendations" not in st.session_state:
        st.session_state.show_recommendations = False

    # 로컬 파일에서 WebP 이미지 로드
    image_path = "/home/jin/level2-3-recsys-finalproject-recsys-07/prototype/logo.png"
    st.image(image_path)
    if st.session_state.show_recommendations:
        apply_custom_css()
        show_recommendations()
        # "처음으로" 버튼 추가
        if st.button("처음으로"):
            st.session_state.show_recommendations = False
            st.rerun()

    else:
        apply_custom_css()
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
                    st.session_state.result_name = get_game_name(
                        st.session_state.result
                    )
                    print(st.session_state.result_name)
                    st.session_state.show_recommendations = True
                    st.rerun()
            else:
                st.error("모든 필드를 입력해주세요.")


if __name__ == "__main__":
    with open("../config/api_key.json", "r") as f:
        conf = json.load(f)
    api_key = conf.get("api_key")
    # main 호출
    main()
