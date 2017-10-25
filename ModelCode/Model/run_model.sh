# train_data_path test_data_path cross_time_test test_ratio good_times  top_num gbdt_pars_default 



#python model_xgb.py tjy/tjy_base_train_10_feature outer/combine_more_data_30_feature 0 1 -1 0 1
#python model_gbdt.py tjy/tjy_base_train_10_feature outer/combine_more_data_30_feature 0 1 -1 0 1

#python model_xgb.py tjy/tjy_base_train_30_feature outer/combine_more_data_30_feature 0 1 -1 0 1
#python model_xgb.py tjy/tjy_base_train_0_30_feature outer/combine_more_data_30_feature 0 1 -1 0 1

#python model_xgb.py tjy/tjy_base_train_10_feature outer/combine_more_data_30_feature 0 1 6 0 1 
#python model_xgb.py tjy/tjy_base_train_10_feature outer/combine_more_data_30_feature 0 1 8 0 1
#python model_xgb.py tjy/tjy_base_train_10_feature outer/combine_more_data_30_feature 0 1 10 0 1

python model_xgb.py tjy/tjy_base_train_10_feature outer/combine_more_data_30_feature 0 0.9 10 0 1
python model_xgb.py tjy/tjy_base_train_10_feature outer/combine_more_data_30_feature 0 0.7 10 0 1
python model_xgb.py tjy/tjy_base_train_10_feature outer/combine_more_data_30_feature 0 0.5 10 0 1


