0. MySQL 설치 및 환경변수 설정
sudo apt-get install mysql-server

##### env 1
mysql_config --cflags
##### env 2
mysql_config --libs

MYSQLCLIENT_CFLAGS=(env 1 출력물)
MYSQLCLIENT_LDFLAGS=(env 2 출력물)

1. airflow 설치
./install_airflow.sh

2. airflow 계정 생성
airflow users create --username OOO --firstname OOO --lastname OOO --role Admin --email OO@OO.com --password OOO

3. airflow 실행
터미널을 2개 켜야 합니다.
터미널 1: airflow webserver ­--port 8080
터미널 2: airflow scheduler

4. airflow 로그인 후 Admin Connections 생성
Steam_DB와 SteamAPI, AppAPI 를 생성해야 합니다.

5. new_id_list.txt 파일 생성
BE/airflow/resource/new_id_list.txt 파일을 생성해야 합니다.

6. DAG 실행
airflow 사이트에서 DAGs 목록에 있는 Daily_update를 활성화하면 실행이 됩니다.