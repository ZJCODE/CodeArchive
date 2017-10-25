import pandas as pd
import math
import tensorflow as tf
from nmf import NMF


if __name__ == '__main__':
    df = pd.read_csv('./temp/ml-100k/u.data', sep='\t', header=None, usecols=[0, 1, 2],
                    names=['userid', 'itemid', 'rating'])
    R = pd.pivot_table(df, values='rating', index=['userid'], columns=['itemid'])
    R.fillna(0, inplace=True)

    ans1 = R[2][1]
    R[2][1] = 0
    ans2 = R[200][940]
    R[200][940] = 0
    ans3 = R[900][931]
    R[900][931] = 0

    nmf = NMF(R.shape[0], R.shape[1])
    
    nmf.sess.run(tf.global_variables_initializer())
    for step in range(50000):
        _, loss, R_pred = nmf.sess.run([nmf.train_op, nmf.loss, nmf.R_pred], {nmf.R:R.values, nmf.lr:0.005})
        if step % 100 == 0:
            print(step, loss)
            print(ans1, '：', R_pred[2][1], ' | ', ans2, ': ', R_pred[200][940],
                  ' | ', ans3, ': ', R_pred[900][931])
