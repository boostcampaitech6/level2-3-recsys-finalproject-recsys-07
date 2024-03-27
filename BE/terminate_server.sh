#!/bin/bash

# 종료할 프로그램의 PID를 찾습니다.
# $(pgrep -f "python ./main.py")

PID= $!

# 프로그램이 실행 중인 경우에만 종료
if [ -n "$PID" ]; then
    echo "프로그램 종료: PID: $PID"
    kill "$PID"
else
    echo "프로그램이 실행 중이 아닙니다."
fi
