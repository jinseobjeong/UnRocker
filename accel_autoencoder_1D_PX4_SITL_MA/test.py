import sys
import argparse

print("hi")
print(sys.version)

parser = argparse.ArgumentParser(description ="사용법 테스트")
parser.add_argument("--env_path" , type = str , help ="타겟값")
parser.add_argument("--script_path" , type = str , help ="input")
parser.add_argument("--target" , type = str , help ="타겟값")
parser.add_argument("--input" , type = str , help ="input")
arg = parser.parse_args()



print("codna 환경 : " , arg.env_path)
print("script 경로 : " , arg.script_path)
print("인풋 : " , arg.input)
print("타겟 : " , arg.target)
#/root/anaconda3/envs/py36/bin/python /home/~~/test.py
