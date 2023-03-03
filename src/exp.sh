#!/bin/bash

python ./run_exp.py --conf_file_path ./config/config_basecnn1.yaml &
sleep 3

python ./run_exp.py --conf_file_path ./config/config_basecnn2.yaml &
sleep 3