Drop Table if Exists tj_tmp.zj_all_sample_0623;
Create Table tj_tmp.zj_all_sample_0623(
    name string,
    mbl_num string,
    id_card string,
    label int,
    loan_dt date,
    source string
)
row format delimited
fields terminated by '\t'
lines terminated by '\n'
stored as textfile;

load data local inpath '/home/zhangjun/Project/multi_loan_2_1/all_sample_0623' overwrite into table tj_tmp.zj_all_sample_0623