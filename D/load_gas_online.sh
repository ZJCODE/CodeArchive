hive -e"

---------建表-------
Drop TABLE if EXISTS beatles_bi_test.gas_online;
CREATE TABLE IF NOT EXISTS beatles_bi_test.gas_online ( 
    store_id bigint,
    city string,
    lng float,
    lat float
)
COMMENT 'gas_online'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
STORED AS TEXTFILE;

--------LOAD DATA语句-----------
--LOCAL是标识符指定本地路径。它是可选的。
--OVERWRITE 是可选的，覆盖表中的数据。

LOAD DATA LOCAL INPATH '/home/zhangjunmichael_i/DataAnalysis/gas_station/evaluation/gas_online.csv' OVERWRITE INTO TABLE beatles_bi_test.gas_online;

"
