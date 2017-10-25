# Pattern

### 文件说明：

---

`Download_Data.py`

从wind数据库下载股票数据和指数数据，并保存为CSV文件

---

`PreProcess.py`

1. Get_Stock_Ts 

   >  返回某只股票的信息

2. Filter_Nan_For_Several_Ts

   > 多个对应的时间序列，如果某个对象某一天没有数据，那么把所有对象的这一天的数据都去除

3. Prepare_Ts_DataFrame

   > 将数据组织成每一列为一只股票信息的格式，方便读取，从而节省检索时间

---

`Calculate_Toolkit.py`

1. sigmoid

   >  转变数据到 (-1,1)，将正数转变到（0,1）

2. Cos_Similarity

   >  计算某一向量与一系列向量的余弦相似性

3. Manhattan_Distance_Similarity

   > 计算某一向量与一系列向量的基于曼哈顿距离的相似性（修正过）/曼哈顿距离

4. Euclidean_Distance_Similarity

   > 计算某一向量与一系列向量的基于欧拉距离的相似性（修正过）/欧拉距离

5. Counter_Sort

   > 对数组计数统计，并从大到小排序

---

`Pattern_Toolkit.py`

```
[暂时不用]
1. Transfer_Array_To_Discrete_Pattern
	 将序列转变为归一化后的离散形式
2. Transfer_Matrix_To_Discrete_Pattern
	将矩阵的每一行转变为归一化后的离散形式
```

1. Extract_Pattern_and_Target

   > 获取历史模式和该模式后的走势序列

2. Resample_Ts

   > 对时间序列进行rolling mean（考虑偏移）

3. Extract_Pattern_and_Target_With_Resample

   > 获取历史模式和该模式后的走势序列(考虑rolling)

4. Pattern_Cluster

   > 对模式序列进行聚类，返回模式和聚类标签

5. Profit_Judge

   > 对预期收益进行评估

---

`Pattern_Cluster_Test.py`

使用以上的函数，对模式进行处理和保存

---

`01Pattern_Extract.py`

对模式进行抽取，抽取后供02反复使用

`02Pattern_Cluster.py	`

对模式进行聚类，聚类后供03模式匹配

`03Pattern_Similarity`

给定序列匹配最佳模式

---



### Pattern_Cluster_Test运行说明：

将这些文件放在同一目录下，然后运行`Pattern_Cluster_Test.py`即可

### 流程说明

Step 1 : 读取数据至`data_df`

Step 2 : 选取准备抽取模式的股票池`code_pool`

Step 3 : 分别输入模式序列及其后续序列的长度值`pattern_len`,`target_len`

Step 4 : 对股票池中的每支股票序列进行模式抽取，得到`pattern_list`和`target_list`

Step 5 : 输入想要聚类的数量`n_clusters`执行聚类操作`Pattern_Cluster`

Step 6 : 构建图片文件输出目录

Step 7 : 针对每一个类进行操作，将同类模式合并，计算同类模式后续的收益情况并绘制相关的图

Step 8 : 将合并后的模式保存到`combine_pattern_pool`，以供之后模式匹配参考

---

### 模式匹配说明

基于提取出来的聚类后的模式库，我们可以进行`Case Based Learning` , 具体思路和`KNN`很相似，简单例子如下

Step 1 ： 对于某一时间序列`ts` , 在 t 时刻 , 我们截取t时刻往前`pattern_len` 长的一段序列`S_t`

Step 2 : 将序列`S_t`与`combine_pattern_pool`中的序列计算相似度可得到最相似的N个模式记为`Set_N`和满足相似度阈值的M个模式`Set_M`

Step 3 : 取以上2个模式集合`Set_N`,`Set_M`的交集，对交集中的模式，查看它们对应的收益情况`combine_profit_mean_pool`,`combine_profit_std_pool`，决定当下的买卖操作，可用于择时，也可用于选股

```
此外，不使用聚类，直接将模式序列作为特征，模式后续序列作为label(按自己的方式做一定的处理)，也可以训练一些General的模型来进行预测
```

