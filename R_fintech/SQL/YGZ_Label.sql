Drop Table If Exists tj_tmp.zjun_ygz_label;
create table tj_tmp.zjun_ygz_label 
row format delimited fields terminated by '\t'
as 

select 

user_mbl_num, loan_dt, case when max(curr_overdue_days)>30 then 1 else 0 end as label, 'ygz' as source

from rtj.success_loan_orders

where loan_dt >= '2017-01-01' and loan_dt < '2017-05-30' 

and curr_overdue_days is not NULL and curr_overdue_days > 0 

and product_name like '%月光足%' 

and product_name not like '%老用户%'

group by user_name, user_mbl_num, loan_dt


hadoop fs -getmerge /user/hive/warehouse/tj_tmp.db/zjun_ygz_label zjun_ygz_label

order_columns = ['mbl_num','tjy_order_id','city_name','bank_name','product_name','application_amount',
            'application_term','application_term_unit','order_stat','approve_amt',
            'conf_loan_amt','order_create_tm','last_update_tm','is_inner_sett']


tjy_order_combine = pd.read_csv('tjy_order_combine','\t',names = order_columns )
label_flie = pd.read_csv('ygz_label',sep='\t',names=['mbl_num','loan_dt','label','source'])
tjy_ygz_order_combine = tjy_order_combine.merge(label_flie[['mbl_num','label','loan_dt','source']],on='mbl_num',how='inner')
tjy_ygz_order_combine.to_csv('tjy_order_combine_with_label_ygz_new',sep='\t',index=False,header=False)
