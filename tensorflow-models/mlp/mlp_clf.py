import tensorflow as tf
import numpy as np
import math


class MLPClassifier:
    def __init__(self, n_in, n_out, hidden_unit_list=[100], sess=tf.Session()):
        """
        Parameters:
        -----------
        n_in: int
            Input dimensions (number of features)
        hidden_unit_list: list or tuple
            List of all hidden units between input and output (e.g. [100, 200, 100])
        n_out: int
            Output dimensions
        sess: object
            tf.Session() object 
        """
        self.n_in = n_in
        self.hidden_unit_list = hidden_unit_list
        self.n_out = n_out
        self.sess = sess
        self._cursor = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_forward_path() 
        self.add_output_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_in])
        self.Y = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.train_flag = tf.placeholder(tf.bool)
        self._cursor = self.X
    # end method add_input_layer


    def add_forward_path(self):
        new_layer = self._cursor
        forward = [self.n_in] + self.hidden_unit_list
        for i in range(1, len(forward)):
            new_layer = self.fc(new_layer, forward[i])
        self._cursor = new_layer
    # end method add_forward_path


    def add_output_layer(self):
        self.logits = tf.layers.dense(self._cursor, self.n_out)
    # end method add_output_layer


    def add_backward_path(self):
        self.lr = tf.placeholder(tf.float32)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                  labels=self.Y))
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits,1), self.Y), tf.float32))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method add_backward_path


    def fc(self, X, fan_out):
        Y = tf.layers.dense(X, fan_out)
        Y = tf.layers.batch_normalization(Y, training=self.train_flag)
        Y = tf.nn.relu(Y)
        Y = tf.nn.dropout(Y, self.keep_prob)
        return Y
    # end method fc (fully-connected)


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, en_exp_decay=True, keep_prob=1.0):
        if val_data is None:
            print("Train %d samples" % len(X) )
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch):
            for local_step, (X_batch, Y_batch) in enumerate(zip(self.gen_batch(X, batch_size),
                                                                self.gen_batch(Y, batch_size))):
                lr = self.adjust_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size)
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                             {self.X: X_batch, self.Y: Y_batch, self.lr: lr,
                                              self.keep_prob:keep_prob, self.train_flag:True})
                global_step += 1
                if local_step % 100 == 0:
                    print ('Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f'
                           %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))
            if val_data is not None:
                # compute validation loss and acc
                val_loss_list, val_acc_list = [], []
                for X_test_batch, Y_test_batch in zip(self.gen_batch(val_data[0], batch_size),
                                                    self.gen_batch(val_data[1], batch_size)):
                    v_loss, v_acc = self.sess.run([self.loss, self.acc],
                                                  {self.X:X_test_batch, self.Y:Y_test_batch,
                                                   self.keep_prob:1.0, self.train_flag:False})
                    val_loss_list.append(v_loss)
                    val_acc_list.append(v_acc)
                val_loss, val_acc = self.list_avg(val_loss_list), self.list_avg(val_acc_list)

            # append to log
            log['loss'].append(loss)
            log['acc'].append(acc)
            if val_data is not None:
                log['val_loss'].append(val_loss)
                log['val_acc'].append(val_acc)

            # verbose
            if val_data is None:
                print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                       "lr: %.4f" % (lr) )
            else:
                print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                       "test_loss: %.4f | test_acc: %.4f |" % (val_loss, val_acc),
                       "lr: %.4f" % (lr) )
        return log
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.logits, {self.X:X_test_batch, self.keep_prob:1.0,
                                                     self.train_flag:False})
            batch_pred_list.append(batch_pred)
        return np.argmax(np.vstack(batch_pred_list), 1)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def adjust_lr(self, en_exp_decay, global_step, n_epoch, len_X, batch_size):
        if en_exp_decay:
            max_lr = 0.003
            min_lr = 0.0001
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len_X/batch_size)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.001
        return lr
    # end method adjust_lr


    def list_avg(self, l):
        return sum(l) / len(l)
# end class