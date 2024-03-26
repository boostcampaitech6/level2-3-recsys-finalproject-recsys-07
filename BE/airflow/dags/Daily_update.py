import re
import logging
import pickle
import math
import torch
import itertools
import pandas as pd
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.operators.dummy import DummyOperator


def read_meta_files():
    # 절대경로 입력하기
    api_key_file_path = (
        "/home/hun/level2-3-recsys-finalproject-recsys-07/BE/resource/API_key.txt"
    )
    new_ids_file_path = (
        "/home/hun/level2-3-recsys-finalproject-recsys-07/BE/resource/new_id_list.txt"
    )

    with open(api_key_file_path, "r", encoding="utf-8") as fr:
        steam_api_key = fr.read()

    with open(new_ids_file_path, "r") as fr:
        new_ids = list(set([line.strip() for line in fr]))

    with open(new_ids_file_path, "w") as fw:
        pass

    return steam_api_key, new_ids


def get_user_profiles(**kwargs):
    ti = kwargs["ti"]
    api_key, new_ids = ti.xcom_pull(task_ids="read_steamID_APIkey")
    user_profile_data = []

    for steam_id in new_ids:
        http_hook = HttpHook(method="GET", http_conn_id="steamAPI")
        endpoint = f"ISteamUser/GetPlayerSummaries/v0002/?key={api_key}&steamids={steam_id}&format=json"

        response = http_hook.run(endpoint)
        if response.status_code == 200:
            data = response.json()
            players = data.get("response", {}).get("players", [])
            if players:
                temp_user_list = [
                    (steam_id, player.get("personaname", "nan"), datetime.now())
                    for player in players
                ]
                user_profile_data.extend(temp_user_list)
            else:
                continue
        else:
            logging.error(f"[get_user_profiles]API CALL ERROR: {steam_id}")
    return user_profile_data


def get_user_games(**kwargs):
    ti = kwargs["ti"]
    api_key, new_ids = ti.xcom_pull(task_ids="read_steamID_APIkey")
    user_game_data = []

    for steam_id in new_ids:
        http_hook = HttpHook(method="GET", http_conn_id="steamAPI")

        endpoint = f"IPlayerService/GetOwnedGames/v0001/?key={api_key}&steamid={steam_id}&format=json&include_appinfo=True&include_played_free_games=True"
        response = http_hook.run(endpoint)
        if response.status_code == 200:
            data = response.json()
            games = data.get("response", {}).get("games", [])
            if games:
                temp_game_list = [
                    (
                        steam_id,
                        game.get("name", "nan"),
                        game.get("appid", "nan"),
                        game.get("playtime_forever", 0),
                        game.get("playtime_2weeks", 0),
                        game.get("img_logo_url", "nan"),
                    )
                    for game in games
                ]
                user_game_data.extend(temp_game_list)
            else:
                continue
        else:
            logging.error(f"[get_user_games]API CALL ERROR: {steam_id}")
    return user_game_data


def get_not_exist_appid_list(**kwargs):
    ti = kwargs["ti"]
    user_game_list = ti.xcom_pull(task_ids="user_games_api_calls")

    mysql_hook = MySqlHook(mysql_conn_id="steam_DB")
    sql = """
        SELECT appid
        FROM app_info;
        """
    conn = mysql_hook.get_conn()
    cur = conn.cursor()
    cur.execute(sql)
    data = cur.fetchall()
    appid_pool = set([temp[0] for temp in data])
    app_infos = pd.DataFrame(
        user_game_list,
        columns=[
            "steamid",
            "name",
            "appid",
            "playtime_forever",
            "playtime_2weeks",
            "img_logo_url",
        ],
    ).astype("str")
    app_infos_newid = app_infos[~app_infos["appid"].isin(appid_pool)]
    app_list = (
        app_infos_newid.loc[:, ["appid", "name"]]
        .drop_duplicates(subset="appid")
        .values.tolist()
    )
    cur.close()
    conn.close()
    return app_list


def get_insert_and_update_list(**kwargs):
    ti = kwargs["ti"]
    user_profile_list = ti.xcom_pull(task_ids="user_profile_api_calls")

    mysql_hook = MySqlHook(mysql_conn_id="steam_DB")
    sql = """
        SELECT steamid
        FROM user_info;
        """
    conn = mysql_hook.get_conn()
    cur = conn.cursor()
    cur.execute(sql)
    data = cur.fetchall()
    steamid_pool = set([temp[0] for temp in data])
    app_infos = pd.DataFrame(
        user_profile_list, columns=["steamid", "personaname", "datatime"]
    ).astype("str")
    insert_user = app_infos[~app_infos["steamid"].isin(steamid_pool)].values.tolist()
    update_user = app_infos[app_infos["steamid"].isin(steamid_pool)].values.tolist()

    cur.close()
    conn.close()
    return insert_user, update_user


def get_platforms(platforms):
    if isinstance(platforms, dict):
        return platforms.values()
    else:
        return [None for i in range(3)]


def get_language(supported_languages):
    if len(supported_languages) < 1:
        return [None for i in range(2)]
    else:
        eng = "english" in supported_languages.lower()
        kor = "korean" in supported_languages.lower()
        return [eng, kor]


def get_multiplayers(categories, genres):
    c_multi = False
    g_multi = False
    if isinstance(categories, list):
        c_ids = [int(cat["id"]) for cat in categories]
        c_multi = bool({1, 37, 39, 24, 44, 27} & set(c_ids))
    if isinstance(genres, list):
        g_ids = [int(gen["id"]) for gen in genres]
        g_multi = 29 in g_ids
    multi = c_multi or g_multi

    return [multi]


def get_boolean_data(categories, genres):
    if isinstance(categories, list):
        c_ids = [int(cat["id"]) for cat in categories]
        # categories
        PvP = bool({49, 36, 37, 47} & set(c_ids))
        Co_op = bool({9, 29, 38, 48} & set(c_ids))
        MMO = 20 in c_ids
        c_booleans = [
            PvP,
            Co_op,
            MMO,
        ]
    else:
        c_booleans = [None for i in range(3)]

    if isinstance(genres, list):
        g_ids = [int(gen["id"]) for gen in genres]
        # genres
        Action = 1 in g_ids
        Adventure = 25 in g_ids
        Indie = 23 in g_ids
        RPG = 3 in g_ids
        Strategy = 2 in g_ids
        Simulation = 28 in g_ids
        Casual = 4 in g_ids
        Sports = 18 in g_ids
        Racing = 9 in g_ids
        Violent = 73 in g_ids
        Gore = 74 in g_ids
        Utilities = bool({53, 55, 57, 51, 60, 59, 52, 56, 58} & set(g_ids))
        Sexual_content = bool({71, 72} & set(g_ids))
        g_booleans = [
            Action,
            Adventure,
            Indie,
            RPG,
            Strategy,
            Simulation,
            Casual,
            Sports,
            Racing,
            Violent,
            Gore,
            Utilities,
            Sexual_content,
        ]
    else:
        g_booleans = [None for i in range(13)]
    return c_booleans + g_booleans


def get_str_data(categories, genres):
    if isinstance(categories, list):
        str_cats = ",".join([cat["description"] for cat in categories])
    else:
        str_cats = None
    if isinstance(genres, list):
        str_genres = ",".join([genre["description"] for genre in genres])
    else:
        str_genres = None
    return [str_cats, str_genres]


def convert_to_int(required_age):
    if isinstance(required_age, str):
        num = re.sub(r"[^0-9]", "", required_age)
        if len(num) > 2:
            age = int(num[:2])
        else:
            age = int(num)
    else:
        age = required_age
    return age


def get_is_adult(required_age):
    if isinstance(required_age, int):
        adult = bool(required_age >= 17)
    else:
        adult = None
    return [adult]


def get_app_add_info(app_id):
    http_hook = HttpHook(method="GET", http_conn_id="appAPI")
    endpoint = f"api/appdetails/?appids={app_id}&l=english"

    response = http_hook.run(endpoint)
    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            logging.error(f"[get_app_add_info]Not JSON Style ERROR: {app_id}")
            return 0

        item = data.get(app_id, {}).get("data", [])
        if item:
            converted_age = convert_to_int(item.get("required_age", None))

            basic_data = [
                item.get("type", None),
                converted_age,
                item.get("is_free", None),
            ]
            is_adult = get_is_adult(converted_age)
            categories = item.get("categories", "nan")
            genres = item.get("genres", "nan")
            platforms = get_platforms(item.get("platforms", "nan"))
            languages = get_language(item.get("supported_languages", "nan"))
            multi = get_multiplayers(categories, genres)
            str_cate_genre = get_str_data(categories, genres)
            boolean_cate_genre = get_boolean_data(categories, genres)
            appid_ = [app_id]
            combined_iter = itertools.chain(
                basic_data,
                is_adult,
                platforms,
                languages,
                multi,
                str_cate_genre,
                boolean_cate_genre,
                appid_,
            )
            return list(combined_iter)
        else:
            [None for i in range(28)]
    else:
        logging.error(f"[get_user_profiles]API CALL ERROR: {app_id}")
        return 0


def insert_user_info_table(**kwargs):
    ti = kwargs["ti"]
    user_profile_list, _ = ti.xcom_pull(task_ids="separation_insert_and_update_ids")

    mysql_hook = MySqlHook(mysql_conn_id="steam_DB")

    sql_profiles = """INSERT INTO `user_info`
                    (steamid, personaname, data_time) 
                    VALUES (%s, %s, %s);"""
    conn = mysql_hook.get_conn()
    cur = conn.cursor()
    for user_profile in user_profile_list:
        cur.execute(sql_profiles, user_profile)
        conn.commit()

    cur.close()
    conn.close()


def update_user_info_table(**kwargs):
    ti = kwargs["ti"]
    _, user_profile_list = ti.xcom_pull(task_ids="separation_insert_and_update_ids")

    mysql_hook = MySqlHook(mysql_conn_id="steam_DB")

    sql_profiles = """UPDATE `user_info`
                    SET data_time=%s, personaname=%s
                    WHERE steamid LIKE %s"""

    conn = mysql_hook.get_conn()
    cur = conn.cursor()
    for user_profile in user_profile_list:
        user = list(reversed(user_profile))
        cur.execute(sql_profiles, user)
        conn.commit()

    cur.close()
    conn.close()


def insert_interactions_table(**kwargs):
    ti = kwargs["ti"]
    user_profile_list, _ = ti.xcom_pull(task_ids="separation_insert_and_update_ids")
    user_id_list = [user[0] for user in user_profile_list]
    user_game_list = ti.xcom_pull(task_ids="user_games_api_calls")
    app_infos = pd.DataFrame(
        user_game_list,
        columns=[
            "steamid",
            "name",
            "appid",
            "playtime_forever",
            "playtime_2weeks",
            "img_logo_url",
        ],
    ).astype("str")
    interaction_list = app_infos.loc[
        :, ["steamid", "appid", "playtime_forever", "playtime_2weeks"]
    ].values.tolist()

    mysql_hook = MySqlHook(mysql_conn_id="steam_DB")

    sql_profiles = """INSERT INTO `user_game_interaction`
                    (user_id, appid, playtime_forever, playtime_2weeks) 
                    VALUES (%s, %s, %s, %s);"""
    conn = mysql_hook.get_conn()
    cur = conn.cursor()

    for interaction in interaction_list:
        if interaction[0] in user_id_list:
            cur.execute(sql_profiles, interaction)
            conn.commit()

    cur.close()
    conn.close()


def update_interactions_table(**kwargs):
    ti = kwargs["ti"]
    _, user_profile_list = ti.xcom_pull(task_ids="separation_insert_and_update_ids")
    user_id_list = [user[0] for user in user_profile_list]
    user_game_list = ti.xcom_pull(task_ids="user_games_api_calls")

    app_infos = pd.DataFrame(
        user_game_list,
        columns=[
            "steamid",
            "name",
            "appid",
            "playtime_forever",
            "playtime_2weeks",
            "img_logo_url",
        ],
    ).astype("str")
    interaction_list = app_infos.loc[
        :, ["steamid", "appid", "playtime_forever", "playtime_2weeks"]
    ].values.tolist()

    mysql_hook = MySqlHook(mysql_conn_id="steam_DB")

    sql_delete = """ DELETE FROM `user_game_interaction`
                    WHERE user_id LIKE %s"""

    sql_profiles = """INSERT INTO `user_game_interaction`
                    (user_id, appid, playtime_forever, playtime_2weeks) 
                    VALUES (%s, %s, %s, %s);"""
    conn = mysql_hook.get_conn()
    cur = conn.cursor()

    for user_id in user_id_list:
        cur.execute(sql_delete, user_id)
        conn.commit()

    for interaction in interaction_list:
        if interaction[0] in user_id_list:
            cur.execute(sql_profiles, interaction)
            conn.commit()

    cur.close()
    conn.close()


def insert_app_info_table(**kwargs):
    ti = kwargs["ti"]
    app_list = ti.xcom_pull(task_ids="get_not_exist_app_ids")
    mysql_hook = MySqlHook(mysql_conn_id="steam_DB")
    # 총 30개 컬럼
    sql_app = """INSERT INTO `app_info` (
                    appid, name, app_type, required_age, is_adult, is_free, on_windows, on_mac, on_linux,
                    English, Korean, multi_player, categories, genres, PvP, Co_op, MMO, Action, Adventure, Indie, 
                    RPG, Strategy, Simulation, Casual, Sports, Racing, Violent, Gore, Utilities, Sexual_content
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
    conn = mysql_hook.get_conn()
    cur = conn.cursor()

    for app in app_list:
        info = get_app_add_info(app[0])
        insert_list = app + info
        cur.execute(sql_app, insert_list)
        conn.commit()
    cur.close()
    conn.close()


def mark_app_use_to_table():
    mysql_hook = MySqlHook(mysql_conn_id="steam_DB")
    conn = mysql_hook.get_conn()
    cur = conn.cursor()
    type_sql = f"""
        UPDATE app_info
        SET `app_use` = CASE
            WHEN app_type LIKE 'game' THEN 1
            ELSE 0
        END;
    """

    genres_sql = f"""
        UPDATE app_info
        SET `app_use` = 0
        WHERE Utilities = 1 OR Sexual_content = 1;
    """

    lang_sql = f"""
        UPDATE app_info
        SET `app_use` = 0
        WHERE English=0 AND Korean=0;
    """

    app_use_sql = """
        UPDATE user_game_interaction A
        JOIN `app_info` B ON A.appid=B.appid
        SET A.app_use = B.app_use
        WHERE A.app_use IS NULL;
    """

    cur.execute(type_sql)
    cur.execute(genres_sql)
    cur.execute(lang_sql)
    cur.execute(app_use_sql)
    conn.commit()
    cur.close()
    conn.close()


def preprocessing_and_training():
    mysql_hook = MySqlHook(mysql_conn_id="steam_DB")
    sql = """
        SELECT user_id, appid, playtime_forever
        FROM user_game_interaction
        WHERE user_use = 1 AND app_use=1;
        """
    conn = mysql_hook.get_conn()
    cur = conn.cursor()
    cur.execute(sql)
    data = cur.fetchall()

    interaction = pd.DataFrame(
        data,
        columns=[
            "steamid",
            "appid",
            "playtime_forever",
        ],
    )

    # Outlier user 제거
    df = interaction[["steamid", "playtime_forever"]].sort_values(
        ["steamid", "playtime_forever"]
    )
    df.reset_index(drop=True, inplace=True)

    n = 5
    rate = 0.01
    outlier = set()

    for i in range(n, len(df) - n):
        user, time = df.loc[i]
        if time < 300 * 60:
            continue
        user_left, time_left = df.loc[i - n]
        user_right, time_right = df.loc[i + n]
        if (
            (user == user_left and user == user_right)
            and time_left > time * (1 - rate)
            and time_right < time * (1 + rate)
        ):
            outlier.add(user)

    interaction["outlier"] = interaction.steamid.apply(
        lambda x: 1 if x in outlier else 0
    )
    interaction = interaction[interaction["outlier"] == 0]

    # Z score 계산
    interaction = interaction[["steamid", "appid", "playtime_forever"]]

    group_df = pd.DataFrame(interaction.groupby("appid").playtime_forever.mean())
    interaction = interaction.merge(
        group_df.rename(columns={"playtime_forever": "mean"}), on="appid", how="left"
    )

    group_df2 = pd.DataFrame(interaction.groupby("appid").playtime_forever.std())
    interaction = interaction.merge(
        group_df2.rename(columns={"playtime_forever": "std"}), on="appid", how="left"
    )

    interaction["z_score"] = (
        interaction["playtime_forever"] - interaction["mean"]
    ) / interaction["std"]

    def z_(row):
        if row["z_score"] > 1.96:
            return row["std"] * 1.96 + row["mean"]
        return row["playtime_forever"]

    interaction["playtime_forever"] = interaction.apply(z_, axis=1)
    interaction["z_score"] = interaction["z_score"].apply(
        lambda z: 1.96 if z > 1.96 else z
    )

    # train
    train(interaction[interaction["std"] > 0])


def labeling(time):
    if time == 0:
        return 0
    else:
        base = 10000 * 60
        return math.log(time, base)


def train(df):
    lambda_ = 100

    df["playtime_forever"] = df["playtime_forever"].apply(labeling)
    df["playtime_forever"] += df["z_score"]
    df["playtime_forever"].apply(lambda x: max(x, 0))

    pivot = df.pivot(
        index="steamid", columns="appid", values="playtime_forever"
    ).fillna(0)

    X = torch.tensor(pivot.values).to(dtype=torch.float).to("cuda")
    G = X.T @ X

    G += torch.eye(G.shape[0]).to("cuda") * lambda_

    P = G.inverse()

    B = P / (-1 * P.diag())
    for i in range(len(B)):
        B[i][i] = 0

    # 절대경로 입력하기
    with open(
        "/home/hun/level2-3-recsys-finalproject-recsys-07/BE/model/B.pickle", "wb"
    ) as f:
        pickle.dump(B, f)
    with open(
        "/home/hun/level2-3-recsys-finalproject-recsys-07/BE/model/app_stat.pickle",
        "wb",
    ) as f:
        df = pd.DataFrame(df.groupby("appid")[["mean", "std"]].max())
        pickle.dump(df, f)


with DAG(
    dag_id="DAILY_UPDATE",
    description="Data collection via API call, Updating DB and Refresh Model Inference",
    schedule_interval="00 0 * * *",
    start_date=datetime(2024, 3, 22, 15, 0, 0),  # 한국 시간으로 매일 자정
    tags=["new_id", "collection", "updating", "training", "making_files"],
) as dag:

    start = DummyOperator(task_id="start")

    read_steamID_APIkey = PythonOperator(
        task_id="read_steamID_APIkey",
        python_callable=read_meta_files,
    )

    user_profile_api_calls = PythonOperator(
        task_id="user_profile_api_calls",
        python_callable=get_user_profiles,
    )

    user_games_api_calls = PythonOperator(
        task_id="user_games_api_calls",
        python_callable=get_user_games,
    )

    get_not_exist_app_ids = PythonOperator(
        task_id="get_not_exist_app_ids",
        python_callable=get_not_exist_appid_list,
    )

    separation_insert_and_update_ids = PythonOperator(
        task_id="separation_insert_and_update_ids",
        python_callable=get_insert_and_update_list,
    )

    insert_user_info = PythonOperator(
        task_id="insert_user_info",
        python_callable=insert_user_info_table,
    )

    update_user_info = PythonOperator(
        task_id="update_user_info",
        python_callable=update_user_info_table,
    )

    insert_interaction = PythonOperator(
        task_id="insert_interaction",
        python_callable=insert_interactions_table,
    )

    update_interactions = PythonOperator(
        task_id="update_interactions",
        python_callable=update_interactions_table,
    )

    insert_app_info = PythonOperator(
        task_id="insert_app_info",
        python_callable=insert_app_info_table,
    )

    mark_app_use = PythonOperator(
        task_id="mark_app_use",
        python_callable=mark_app_use_to_table,
    )

    mark_user_use = MySqlOperator(
        task_id="mark_user_use",
        mysql_conn_id="steam_DB",
        sql="""
            UPDATE user_game_interaction AS A
            JOIN (
                SELECT user_id
                FROM user_game_interaction
                WHERE app_use = 1
                GROUP BY user_id
                HAVING SUM(playtime_forever)=0 OR SUM(playtime_2weeks)>16128 
            ) AS B ON A.user_id = B.user_id
            SET A.user_use = 0;
        """,
    )

    file_generation = PythonOperator(
        task_id="file_generation",
        python_callable=preprocessing_and_training,
    )

    finish = DummyOperator(task_id="finish")

    (
        start
        >> read_steamID_APIkey
        >> user_profile_api_calls
        >> user_games_api_calls
        >> get_not_exist_app_ids
        >> separation_insert_and_update_ids
        >> insert_user_info
        >> update_user_info
        >> insert_interaction
        >> update_interactions
        >> insert_app_info
        >> mark_app_use
        >> mark_user_use
        >> file_generation
        >> finish
    )
