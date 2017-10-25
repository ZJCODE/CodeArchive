

Drop Table if Exists tj_tmp.zj_test;
Create Table tj_tmp.zj_test(
    id string,
    num int,
    word string,
    test int
)
row format delimited
fields terminated by '\t'
lines terminated by '\n'
stored as textfile;

load data local inpath '/home/zhangjun/test' overwrite into table tj_tmp.zj_test



# ---------------------------------------------------

Drop Table If Exists tj_tmp.tjy_order_combine;
create table tj_tmp.tjy_order_combine 
row format delimited fields terminated by '\t'
as 
select  
    user_mbl_num,
    concat_ws(',',collect_list(regexp_replace(case when tjy_order_id is null then 'nan' else cast(tjy_order_id as string) end ,',',''))) as tjy_order_id,
    concat_ws(',',collect_list(regexp_replace(case when city_name is null then 'nan' else cast(city_name as string) end ,',',''))) as city_name,
    concat_ws(',',collect_list(regexp_replace(case when bank_name is null then 'nan' else cast(bank_name as string) end ,',',''))) as bank_name,
    concat_ws(',',collect_list(regexp_replace(case when product_name is null then 'nan' else cast(product_name as string) end ,',',''))) as product_name,
    concat_ws(',',collect_list(regexp_replace(case when application_amount is null then 'nan' else cast(application_amount as string) end ,',',''))) as application_amount,
    concat_ws(',',collect_list(regexp_replace(case when application_term is null then 'nan' else cast(application_term as string) end ,',',''))) as application_term,
    concat_ws(',',collect_list(regexp_replace(case when application_term_unit is null then 'nan' else cast(application_term_unit as string) end ,',',''))) as application_term_unit,
    concat_ws(',',collect_list(regexp_replace(case when order_stat is null then 'nan' else cast(order_stat as string) end ,',',''))) as order_stat,
    concat_ws(',',collect_list(regexp_replace(case when approve_amt is null then 'nan' else cast(approve_amt as string) end ,',',''))) as approve_amt,
    concat_ws(',',collect_list(regexp_replace(case when conf_loan_amt is null then 'nan' else cast(conf_loan_amt as string) end ,',',''))) as conf_loan_amt,
    concat_ws(',',collect_list(regexp_replace(case when create_tm is null then 'nan' else cast(create_tm as string) end ,',',''))) as order_create_tm,
    concat_ws(',',collect_list(regexp_replace(case when last_update_tm is null then 'nan' else cast(last_update_tm as string) end ,',',''))) as last_update_tm,
    concat_ws(',',collect_list(regexp_replace(case when is_inner_sett is null then 'nan' else cast(is_inner_sett as string) end ,',',''))) as is_inner_sett
from tjtjy.t04_tjy_order
group by user_mbl_num




# ---------------------------------------------------

Drop Table If Exists tj_tmp.tjy_order_feedback_combine;
create table tj_tmp.tjy_order_feedback_combine 
row format delimited fields terminated by '\t'
as 

select  
user_mbl_num,
concat_ws(',',collect_list(regexp_replace(cast(approve_conclusion as string),',',''))) as approve_conclusion,
concat_ws(',',collect_list(regexp_replace(cast(application_amount as string),',',''))) as application_amount,
concat_ws(',',collect_list(regexp_replace(cast(application_term as string),',',''))) as application_term,
concat_ws(',',collect_list(regexp_replace(cast(application_term_unit as string),',',''))) as application_term_unit,
concat_ws(',',collect_list(regexp_replace(cast(extra_fee as string),',',''))) as extra_fee,
concat_ws(',',collect_list(regexp_replace(cast(per_repay_fee as string),',',''))) as per_repay_fee,
concat_ws(',',collect_list(regexp_replace(cast(bind_card_stat as string),',',''))) as bind_card_stat,
concat_ws(',',collect_list(regexp_replace(cast(create_tm as string),',',''))) as create_tm


from 

(
select tjy_order_id,approve_conclusion,application_amount,
        application_term,application_term_unit,extra_fee,per_repay_fee, bind_card_stat
    from tjtjy.t04_tjy_orders_bank_feedback 
)feedback
left join
(
    select  tjy_order_id,user_mbl_num,create_tm from tjtjy.t04_tjy_order 
)order
on feedback.tjy_order_id = order.tjy_order_id

where user_mbl_num is not null
group by user_mbl_num

# ---------------------------------------------------




Drop Table If Exists tj_tmp.tjy_repayment_plan_combine;
create table tj_tmp.tjy_repayment_plan_combine 
row format delimited fields terminated by '\t'
as 


select 

user_mbl_num,
concat_ws(',',collect_list(regexp_replace(cast(repay_per_num as string),',',''))) as repay_per_num,
concat_ws(',',collect_list(regexp_replace(cast(tot_repay_amt as string),',',''))) as tot_repay_amt,
concat_ws(',',collect_list(regexp_replace(cast(repay_dt as string),',',''))) as repay_dt,
concat_ws(',',collect_list(regexp_replace(cast(succ_repay_dt as string),',',''))) as succ_repay_dt,
concat_ws(',',collect_list(regexp_replace(cast(is_overdue as string),',',''))) as is_overdue,
concat_ws(',',collect_list(regexp_replace(cast(create_tm as string),',',''))) as create_tm


from 

(
select tjy_order_id,repay_per_num, tot_repay_amt,repay_dt,succ_repay_dt,is_overdue,
        row_number() over(partition by tjy_order_id order by succ_repay_dt desc) as succ_repay_number
from tjtjy.t02_tjy_repayment_plan
) repay

left join 

(
select tjy_order_id,repay_mode,application_amount, application_term

from tjtjy.t04_tjy_orders_bank_feedback
)feedback

on repay.tjy_order_id = feedback.tjy_order_id



(
select  tjy_order_id,user_mbl_num,create_tm from tjtjy.t04_tjy_order 
)order 
on repay.tjy_order_id = order.tjy_order_id

where user_mbl_num is not null
group by user_mbl_num


#-----------------------------------------------------


Drop Table If Exists tj_tmp.stat_log_combine;
create table tj_tmp.stat_log_combine 
row format delimited fields terminated by '\t'
as 

select 

user_mbl_num,
concat_ws(',',collect_list(regexp_replace(cast(stat as string),',',''))) as stat,
concat_ws(',',collect_list(regexp_replace(cast(tjy_order_id as string),',',''))) as tjy_order_id,
concat_ws(',',collect_list(regexp_replace(cast(log_create_dt as string),',',''))) as log_create_dt

from 

(
select tjy_order_id,stat,create_dt as log_create_dt
from tjtjy.t04_tjy_order_stat_log
)log 

left join 

(
select  tjy_order_id,user_mbl_num from tjtjy.t04_tjy_order 
)order 

on log.tjy_order_id = order.tjy_order_id

where user_mbl_num is not null
group by user_mbl_num




