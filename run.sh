#!/usr/bin/bash
source proj_env/bin/activate

file_name=$1
dump=$2

python3 ./eval/test_one_solution.py --jsonl_path "$dump/cache/$file_name.jsonl" --dump $dump
