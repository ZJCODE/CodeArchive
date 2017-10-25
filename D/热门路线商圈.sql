

# 城市： ('北京','杭州','成都','哈尔滨','深圳')
# 将所有的城市替换成自己想看的那几个城市
# 运行完之后将文件放在统一目录下运行拆分分析.py

hive -e"
select 
record_day,
city_name,
today_call_incity_o_cnt,
today_call_crosscity_o_cnt,
today_call_incity_o_cnt/today_call_crosscity_o_cnt as city_cross_city_ratio
from beatles_dm.dm_beatles_daily_order_part2
where concat_ws('-',year,month,day) between '2017-04-04' and '2017-06-04'
and city_name in ('北京','杭州','成都','哈尔滨','深圳')
">市内跨城占比.csv


hive -e"

select

a.record_day as record_day,
a.city_name as city_name,
avg(today_rob_incity_d_cnt)/avg(today_call_incity_p_cnt) as driver_passenger_ratio, -- 市内司乘比
avg(today_call_incity_o_cnt) as today_call_incity_o_cnt, -- 市内日呼叫订单
avg(today_incity_reply_rate) as today_incity_reply_rate, -- 市内应答率
avg(today_arrive_incity_rate) as today_arrive_incity_rate, -- 市内成功送达率
avg(today_pay_incity_avgfee) as today_pay_incity_avgfee, -- 市内平均订单价格
avg(today_pay_incity_median_distance) as today_pay_incity_median_distance, -- 市内平均订单距离
avg(today_call_incity_cancel_beforereply_o_cnt)/avg(today_call_incity_o_cnt) as city_cancal_rate -- 市内应答前取消率

from
(
    select 
    record_day,city_name,
    today_call_incity_p_cnt,  -- 今日市内呼叫乘客数  
    today_call_incity_o_cnt, -- 今日市内订单量 
    today_incity_reply_rate, -- 今日市内订单今日应答率 
    today_call_incity_cancel_beforereply_o_cnt, -- 今日市内订单今日应答前取消量
    today_pay_incity_avgfee, -- 今日支付市内订单平均单价
    today_arrive_incity_rate, -- 市内出行订单今日成交率
    today_pay_incity_median_distance -- 今日支付市内订单里程中位数

    from beatles_dm.dm_beatles_daily_order_part2
    where concat_ws('-',year,month,day) between '2017-04-04' and '2017-06-04'
    and city_name in ('北京','杭州','成都','哈尔滨','深圳')
)a

left join 

(
    select 
    record_day,city_name,
    today_rob_incity_d_cnt, -- 今日抢市内订单车主数
    today_rob_cross_d_cnt -- 今日抢跨城订单车主数
    from beatles_dm.dm_beatles_daily_driver
    where concat_ws('-',year,month,day) between '2017-04-04' and '2017-06-04'
    and city_name in ('北京','杭州','成都','哈尔滨','深圳')
)b 

on a.city_name = b.city_name


group by a.record_day , a.city_name

">市内.csv

------------------------------------------------------------------------------

hive -e"

select

a.record_day as record_day,
a.city_name as city_name,

avg(today_rob_cross_d_cnt)/avg(today_call_crosscity_p_cnt) as driver_passenger_ratio, -- 跨城司乘比
avg(today_call_crosscity_o_cnt) as today_call_crosscity_o_cnt, -- 跨城日呼叫订单
avg(today_crosscity_reply_rate) as today_crosscity_reply_rate, -- 跨城应答率
avg(today_arrive_crosscity_rate) as today_arrive_crosscity_rate, -- 跨城成功送达率
avg(today_pay_crosscity_avgfee) as today_pay_crosscity_avgfee, -- 跨城平均订单价格
avg(today_call_crosscity_cancel_beforereply_o_cnt)/avg(today_call_crosscity_o_cnt) as cross_city_cancal_rate -- 跨城应答前取消率

from
(
    select 
    record_day,city_name,
    today_call_crosscity_p_cnt,  -- 今日跨城呼叫乘客数   
    today_call_crosscity_o_cnt, -- 今日跨城订单量  
    today_crosscity_reply_rate, -- 今日跨城订单今日应答率  
    today_call_crosscity_cancel_beforereply_o_cnt, -- 今日跨城订单今日应答前取消量
    today_pay_crosscity_avgfee, -- 今日支付跨城订单平均单价
    today_arrive_crosscity_rate, -- 跨城出行订单今日成交率
    today_call_crosscity_mid_distance -- 今日支付跨城订单里程中位数

    from beatles_dm.dm_beatles_daily_order_part2
    where concat_ws('-',year,month,day) between '2017-04-04' and '2017-06-04'
    and city_name in ('北京','杭州','成都','哈尔滨','深圳')
)a

left join 

(
    select 
    record_day,city_name,
    today_rob_incity_d_cnt, -- 今日抢市内订单车主数
    today_rob_cross_d_cnt -- 今日抢跨城订单车主数
    from beatles_dm.dm_beatles_daily_driver
    where concat_ws('-',year,month,day) between '2017-04-04' and '2017-06-04'
    and city_name in ('北京','杭州','成都','哈尔滨','深圳')
)b 

on a.city_name = b.city_name


group by a.record_day , a.city_name

">跨城.csv

------------------------------------------------------------------------------

hive -e"

select

road,
count(order_id) as order_cnt,
avg(passenger_sum_num) as passenger_sum_num_avg,
avg(navi_distance) as navi_distance

from 

(

    select 
    order_id,concat_ws('-',from_city_name,to_city_name) as road ,passenger_sum_num/100 as passenger_sum_num,navi_distance/1000 as navi_distance
    from  beatles_dwd.dwd_order_create_d
    where pt between '2017-04-04' and '2017-06-04'
    and status in (2,3,4,5,6,13,31)
    and (from_city_name in ('北京','杭州','成都','哈尔滨','深圳') or to_city_name in ('北京','杭州','成都','哈尔滨','深圳'))
    and from_city_name != to_city_name
    and is_test != 1
   
)a
group by road
order by order_cnt desc 

">跨城热门路线.csv


------------------------------------------------------------------------------


hive -e"
select
from_city_name as city_name,
from_to_name,
order_cnt,
passenger_sum_num_avg

from 
(
    select

        from_to_loc,
        from_city_name,
        count(order_id) as order_cnt,
        avg(passenger_sum_num) as passenger_sum_num_avg
    from 
    (
        select 

        order_id,from_city_name,to_city_name,passenger_sum_num/100 as passenger_sum_num,passenger_id,
        concat_ws('-',cast(round(from_lng,2) as string),cast(round(from_lat,2) as string),cast(round(to_lng,2) as string),cast(round(to_lat,2) as string)) as from_to_loc

        from  beatles_dwd.dwd_order_create_d
        where pt between '2017-04-04' and '2017-06-04'
        and status in (2,3,4,5,6,13,31)
        and (from_city_name in ('北京','杭州','成都','哈尔滨','深圳') and to_city_name in ('北京','杭州','成都','哈尔滨','深圳'))
        and from_city_name = to_city_name
        and is_test != 1
    )c 


    group by from_to_loc,from_city_name
)d 


left join
-- 热门地区具体地址获取（基于截断的经纬度地址）
(

select

from_to_loc,from_to_name

from 
(
    select 
        from_to_loc,
        from_to_name,
        ROW_NUMBER() OVER(PARTITION BY from_to_loc ORDER BY order_cnt desc ) AS loc

    from 
    (

        select

        from_to_loc,from_to_name,
        count(order_id) as order_cnt 

        from 

        (
            select 

            order_id,from_city_name,to_city_name,passenger_sum_num/100 as passenger_sum_num,
            concat_ws('-',cast(round(from_lng,2) as string),cast(round(from_lat,2) as string),cast(round(to_lng,2) as string),cast(round(to_lat,2) as string)) as from_to_loc,
            concat_ws('|->|',from_name,to_name) as from_to_name

        from  beatles_dwd.dwd_order_create_d
        where pt between '2017-04-04' and '2017-06-04'
        and status in (2,3,4,5,6,13,31)
        and (from_city_name in ('北京','杭州','成都','哈尔滨','深圳') and to_city_name in ('北京','杭州','成都','哈尔滨','深圳'))
        and from_city_name = to_city_name
        and is_test != 1
        )a 

        group by from_to_loc,from_to_name
    )b
)e

where loc ==1

)place_name

on d.from_to_loc = place_name.from_to_loc

order by order_cnt desc
">市内热门路线.csv
