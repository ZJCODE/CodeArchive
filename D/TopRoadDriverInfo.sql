

hive -e"

select

c.driver_id as driver_id, -- 司机ID
c.user_rode as user_rode,  -- 司机走得最多的线路
c.phone as phone, -- 司机手机号
c.cnt as cnt, -- 司机在这段时间内在该线路的行程数量
c.city_name as city_name -- 城市

from 

(
    select
    driver_id,
    user_rode,
    phone,
    cnt,
    from_city_name as city_name,
    ROW_NUMBER() OVER(PARTITION BY driver_id ORDER BY cnt desc ) AS loc
    from 
    (
    select 
    driver_id,
    from_city_name,
    concat(cast(driver_id as string),'-|',from_name,'->',to_name) as user_rode,
    count(driver_id) as cnt
    from  beatles_dwd.dwd_order_create_d
    where hour(setup_time) in (7,8,9,17,18,19,20) -- 通勤时间限制
    and pt between '2017-05-04' and '2017-06-06'
    and from_city_name = '北京'
    and status in (2,3,4,5,6,13,31)
    and driver_id > 0
    and pmod(datediff(pt, '2012-01-01'), 7) not in (0,1) -- 工作日限制
    group by driver_id,concat(cast(driver_id as string),'-|',from_name,'->',to_name),from_area_id

    )a 

    left join 

    (
    select 
    user_id,
    phone
    from 
    beatles_dwd.dwd_driver_vehicle_info
    )b 

    on a.driver_id = b.user_id
)c 

where loc = 1
and int(c.phone/100000000) != 110 -- 去掉测试账号
order by cnt desc
limit 3000

">TopRoadDriverInfo.csv


