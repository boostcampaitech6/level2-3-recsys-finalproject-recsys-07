1. model 폴더 내에 'DB_interaction.csv' 파일을 준비해주세요.
2. BE/model 폴더에서 python EASE_train.py를 실행해주세요.
3. ../config 폴더에 api_key.json 및 DB_config.json 파일을 준비해주세요.
# (양식)
{
    "api_key": "API KEY 입력"
}

{
    "host": "DB 네트워크 주소",
    "port": 포트 번호,
    "user": "DB 아이디",
    "password": "DB password",
    "db": "steam_data" # 사용할 DB 이름
}
4. BE 폴더에서 python main.py를 통해 서버를 실행해주세요.