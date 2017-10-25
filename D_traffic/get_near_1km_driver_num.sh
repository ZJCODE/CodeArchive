#!/bin/sh
source /etc/profile
source ~/.bashrc
source dateparam.sh


V_DATE=$1

if [ -z ${V_DATE} ];then
        V_DATE=`date +%Y-%m-%d`
else
    V_DATE=`date --date="$V_DATE+1 day" +%Y-%m-%d`
fi

V_PARYEAR=`date --date="$V_DATE-1 day" +%Y`
V_PARMONTH=`date --date="$V_DATE-1 day" +%m`
V_PARDAY=`date --date="$V_DATE-1 day" +%d`
V_PARWEEK=`date --date="$V_DATE-1 day" +%V`
V_PARTODAY=`date --date="$V_DATE-1 day" +%Y%m%d`
V_YESTERDAY=`date --date="$V_DATE-1 day" +%Y-%m-%d`
V_BEGINDAY=`date --date="$V_DATE-7 day" +%Y-%m-%d`
V_LASTEND=`date --date="$V_DATE-8 day" +%Y-%m-%d`
V_LASTBEGIN=`date --date="$V_DATE-14 day" +%Y-%m-%d`
V_90DAYS=`date --date="$V_DATE-90 day" +%Y-%m-%d`
V_180DAYS=`date --date="$V_DATE-180 day" +%Y-%m-%d`
echo $V_BEGINDAY
echo $V_YESTERDAY
echo $V_LASTBEGIN
echo $V_LASTEND
echo $V_PARYEAR
echo $V_PARMONTH
echo $V_PARDAY
echo $V_PARWEEK




hive -e"
select 

city_name, -- 城市
day, -- 日期 
count(distinct case when ( 6378.137*acos(sin(lat/57.2958) * sin(from_lat/57.2958) + cos(lat/57.2958) * cos(from_lat/57.2958) * cos(lng/57.2958 - from_lng/57.2958)) <=1 ) or ( 6378.137*acos(sin(lat/57.2958) * sin(to_lat/57.2958) + cos(lat/57.2958) * cos(to_lat/57.2958) * cos(lng/57.2958 - to_lng/57.2958)) <=1 ) then driver_id end) as gas_near_driver_rob_cnt -- 该城市加油站周边一公里所有司机数

from
(
select 
concat_ws('-',year,month,day) as day,
driver_id,
from_city_id,
from_lng,from_lat,
to_lng,to_lat,
1 as tag
from beatles_dwd.dwd_order_rob_succ_d
where concat_ws('-',year,month,day) between'${s_forward_10day}' and '${s_day}'
)a 

join

(
    select 
        city_id,
        city_name
    from  beatles_dw.dim_city 
    where city_name in ('东莞','佛山','廊坊','长春','石家庄','张家口','南京','昆明','济南','温州') -- 查看的几个城市名称，及加油站地址列表中的城市
)b 

on a.from_city_id = b.city_id

join 

(

select
city , lng,lat , 1 as tag 
from beatles_bi_test.gas_online

)c 

on a.tag = c.tag and b.city_name = c.city

group by city_name,day
">near.csv
