#!/bin/bash

# 파이썬 버전 추출
PYTHON_VERSION=$(python3 --version | cut -d" " -f2 | cut -d"." -f1,2)

# Airflow 버전 설정
AIRFLOW_VERSION=2.6.3

# 제약 사항 URL 설정
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Airflow 설치
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

#Airflow directory 변수 저장
export AIRFLOW_HOME=`pwd`

#변수 저장 여부 확인
echo $AIRFLOW_HOME

#Airflow db 초기화
airflow db init

pip install -U apache-airflow-providers-common-sql
pip install -U mysqlclient
pip install -U mysql-connector-python
pip install -U apache-airflow-providers-mysql

pip install pyarrow