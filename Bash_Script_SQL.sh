#!/bin/sh

sleep 30

if ! mysql -h my-db -u root -e 'use IPL'; then
  echo "IPL DOES NOT exists"
    mysql -h my-db -u root -e "CREATE DATABASE IF NOT EXISTS IPL"
    mysql -h my-db -u root IPL < /scripts/IPL.sql
    #mysql -h my-db -u root IPL < /scripts/small_ipl.sql
else
  echo "IPL DOES exists"
fi

echo "Calling table_gen_sql.sql"
mysql -h my-db -u root IPL < /scripts/table_gen_sql.sql

#mysql -h my-db -u root IPL -e '
#  SELECT * FROM teams;' > /Data_Files/teams.csv
mysql -h my-db -u root IPL -e '
  SELECT * FROM Final_diff_stats;' > /Output-File/final_ipl_stats.csv

# Calling the Python script to generate Model
python3 ./scripts/IPL_project.py
#python3 ./scripts/try_sql_conn.py

