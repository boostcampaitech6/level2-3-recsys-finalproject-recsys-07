1. airflow 폴더 내에서 install_airflow.sh 로 airflow를 설치합니다.
3. 터미널 창을 2개 켜 airflow webserver ­--port 8080 와 airflow scheduler 를 실행합니다.
4. Admin Connections에서 Steam_DB와 SteamAPI, AppAPI 를 생성해주세요. (자세한 방법은 추후 보완)
5. resource 폴더 내에 API_key.txt, new_id_list.txt 파일을 준비해주세요.
6. DB_upload.py 파일의 read_meta_files 함수에 텍스트 파일의 절대경로를 입력합니다.
7. DAG를 실행할 수 있습니다.(아직 시간 설정은 하지 않았습니다.)