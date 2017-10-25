rm ../data/graph_user_*dim*
rm ../data/loc_node*dim*
rm ../data/user_vec*

cp ../data/loc_start_end_pair LINE
mv ../data/graph_user_end_places_pair LINE
mv ../data/graph_user_places_pair LINE
mv ../data/graph_user_start_places_pair LINE



cd LINE
sh get_node_vec.sh $1
cp loc_node_vec_order_combine* ../../data
cp graph_user_*dim* ../../data
rm loc*
rm graph_user_*

