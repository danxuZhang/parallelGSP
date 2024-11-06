#!/usr/bin/env bash


"Running SPMF on 20% FIFA dataset ..."

java -jar /app/spmf.jar run GSP /app/data/FIFA.txt output25_spmf.txt 25%

python3 pgsp.py --min_support_ratio 0.25 /app/data/FIFA.txt output25_pgsq.txt


echo "Running SPMF on 15% FIFA dataset ..."

java -jar /app/spmf.jar run GSP /app/data/FIFA.txt output15_spmf.txt 15%

python3 pgsp.py --min_support_ratio 0.15 /app/data/FIFA.txt output15_pgsq.txt


