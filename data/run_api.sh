#!/bin/bash
# API 호출을 담당하는 쉘 스크립트입니다.

# ../config/api_key.txt 파일에서 API 키를 읽어와서 api_key 변수에 담습니다.
api_key=$(<../config/api_key.txt)

# ../config/api_call.config 파일에서 start_i 를 읽어와서 api_key 변수에 담습니다.
source ../config/api_call.config

# api key를 파이썬에 인자로 제공합니다.
nohup python api_call.py --version="${version}" --api_key="${api_key}" --start_i="${start_i}" &

# nohup.out 파일을 실시간으로 출력합니다.
tail -f nohup.out