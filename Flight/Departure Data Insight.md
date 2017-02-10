# Departure Data Insight

- 同一航班的起飞时间基本固定

  ```
  set(Departure[Departure.flight_ID == 'CZ305'].flight_time)
  Out[228]: 
  {Timestamp('2016-09-11 00:30:00'),
   Timestamp('2016-09-12 00:30:00'),
   Timestamp('2016-09-13 00:30:00'),
   Timestamp('2016-09-13 01:30:00'),
   Timestamp('2016-09-14 00:30:00'),
   Timestamp('2016-09-15 00:30:00')}
  ```

  - 有的航班没有起飞时间

    ```
    Departure[Departure.flight_ID == 'EK205']
    Out[233]: 
            passenger_ID2 flight_ID flight_time        checkin_time flight_Type
    112193    177379213.0     EK205         NaT 2016-09-11 21:07:00          EK
    121261    177384911.0     EK205         NaT 2016-09-11 21:21:00          EK
    208445    178641481.0     EK205         NaT 2016-09-13 20:50:00          EK
    ```

    ​

- 可以统计出每个10Min中起飞的航班数量序列



航班信息与出发位置关联

