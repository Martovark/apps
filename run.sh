#!/usr/bin/bash

file_name=$1
python3 ./eval/test_one_solution.py --jsonl_path "dump/cache/$file_name.jsonl" # --debug