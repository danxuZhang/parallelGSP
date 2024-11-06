#!/usr/bin/env bash


echo "Running SPMF on 20% FIFA dataset ..."

time java -jar /app/spmf.jar run GSP /app/data/FIFA.txt output25_spmf.txt 25%

time python3 pgsp.py --min_support_ratio 0.25 FIFA.txt output25_pgsq.txt  


echo "Running SPMF on 15% FIFA dataset ..."

time java -jar /app/spmf.jar run GSP /app/data/FIFA.txt output15_spmf.txt 15%

time python3 pgsp.py --min_support_ratio 0.25 FIFA.txt output15_pgsq.txt  

