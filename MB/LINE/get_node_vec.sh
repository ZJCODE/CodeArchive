
./line -train loc_start_end_pair  -output loc_node_vec_order_1_dim_$1 -size $1 -order 1 -samples 10
./line -train loc_start_end_pair  -output loc_node_vec_order_2_dim_$1 -size $1 -order 2 -samples 10
python combine.py loc_node_vec_order_1_dim_$1 loc_node_vec_order_2_dim_$1

./line -train graph_user_places_pair  -output graph_user_whole_start_places_vec_order_2_dim_$1 -size $1 -order 2 -samples 10
./line -train graph_user_start_places_pair  -output graph_user_start_places_vec_order_2_dim_$1 -size $1 -order 2 -samples 10
./line -train graph_user_end_places_pair  -output graph_user_end_places_vec_order_2_dim_$1 -size $1 -order 2 -samples 10
