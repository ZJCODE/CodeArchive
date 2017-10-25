import numpy as np
import json
output = open('tjy_order_combine_new','w')

with open('tjy_order_combine','r') as f:
    for line in f.readlines():
        line_list = line.rstrip().split('\t')
        new_line = line_list[0] + '\t' + json.dumps({'order':np.array([x.split(',') for x in line_list[1:]]).T.tolist()}) + '\n'
        output.write(new_line)
'''
tjy_order_combine_columns = ['mbl_num','tjy_order_id','city_name','bank_name','product_name','application_amount',
            'application_term','application_term_unit','order_stat','approve_amt',
            'conf_loan_amt','order_create_tm','last_update_tm','is_inner_sett']
'''