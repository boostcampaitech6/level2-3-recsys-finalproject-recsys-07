#!/bin/bash
# API 호출을 담당하는 쉘 스크립트입니다.

# jq를 설치해야 할 수 있습니다. apt-get install jq
# ../config/api_key.txt 파일에서 API 키를 읽어와서 api_key 변수에 담습니다.
api_key=$(jq -r '.api_key' ../config/api_key.json)

# ../config/api_call.config 파일에서 start_i 를 읽어와서 start_i 변수에 담습니다.
source ../config/api_call.config

{ 
  # api key를 파이썬에 인자로 제공합니다.
  nohup python api_call.py --version="${version}" --api_key="${api_key}" --start_i="${start_i}" &
  # python api_call.py 프로세스 번호를 저장합니다.
  PID=$!
  # api_call.py가 완료되기를 기다립니다.
  wait $PID
  echo $PID
  # 'nohup.out'에서 마지막 'Stop the execution' 메시지 추출
  # tail로 마지막 줄을 가져온 후, awk를 사용해 숫자만 추출
  last_i_value=$(tail -n 1 nohup.out | awk '{print $NF}')
  # 다음 index로 stride만큼 증가
  last_i_value=$((last_i_value + 800))
  # 캡처된 i 값을 사용하여 api_call.config 업데이트
  if [[ ! -z "$last_i_value" ]]; then
    sed -i "s/start_i=.*/start_i=${last_i_value}/" ../config/api_call.config
  fi
} &

# nohup.out 파일을 실시간으로 출력합니다.
sleep 3
tail -f nohup.out

