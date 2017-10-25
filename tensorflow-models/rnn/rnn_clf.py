import tensorflow as tf
import numpy as np
import math
import sklearn


class RNNClassifier:
    def __init__(self, n_in, n_seq, n_out, cell_size=128, n_layer=1, stateful=False, sess=tf.Session()):
        """
        Parameters:
        -----------
        n_in: int
            Input dimensions
        n_step: int
            Number of time steps
        cell_size: int
            Number of units in the rnn cell
        n_out: int
            Output dimensions
        n_layer: int
            Number of layers of stacked rnn cells
        sess: object
            tf.Session() object
        stateful: boolean
            If true, the final state for each batch will be used as the initial state for the next batch 
        """
        self.n_in = n_in
        self.n_seq = n_seq
        self.cell_size = cell_size
        self.n_out = n_out
        self.n_layer = n_layer
        self.sess = sess
        self.stateful = stateful
        self._cursor = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_lstm_cells()
        self.add_dynamic_rnn()
        self.add_output_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_seq, self.n_in])
        self.Y = tf.placeholder(tf.int64, [None])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.in_keep_prob = tf.placeholder(tf.float32)
        self.out_keep_prob = tf.placeholder(tf.float32)
        self._cursor = self.X
    # end method add_input_layer


    def add_lstm_cells(self):
        def cell():
            cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, initializer=tf.orthogonal_initializer())
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.in_keep_prob, self.out_keep_prob)
            return cell
        self.cells = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.n_layer)])
    # end method add_rnn_cells


    def add_dynamic_rnn(self):      
        self.init_state = self.cells.zero_state(self.batch_size, tf.float32)        
        self._cursor, self.final_state = tf.nn.dynamic_rnn(self.cells, self._cursor,
                                                           initial_state=self.init_state,
                                                           time_major=False)
    # end method add_dynamic_rnn

    def add_output_layer(self):
        # (batch, n_step, n_hidden) -> (n_step, batch, n_hidden) -> n_step * [(batch, n_hidden)]
        time_major = tf.unstack(tf.transpose(self._cursor, [1,0,2]))
        self.logits = tf.layers.dense(time_major[-1], self.n_out)
    # end method add_output_layer


    def add_backward_path(self):
        self.lr = tf.placeholder(tf.float32)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                  labels=self.Y))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, axis=1),
                                                   self.Y), tf.float32))
    # end method add_backward_path


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, en_exp_decay=True, en_shuffle=True, 
            keep_prob_tuple=(1.0,1.0)):
        if val_data is None:
            print("Train %d samples" % len(X) )
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch): # batch training
            if en_shuffle:
                X, Y = sklearn.utils.shuffle(X, Y)
            next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})

            for local_step, (X_batch, Y_batch) in enumerate(zip(self.gen_batch(X, batch_size),
                                                                self.gen_batch(Y, batch_size))):
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size)
                if (self.stateful) and (len(X_batch) == batch_size):
                    _, next_state, loss, acc = self.sess.run([self.train_op, self.final_state, self.loss, self.acc],
                                                             {self.X:X_batch, self.Y:Y_batch, self.lr:lr,
                                                              self.in_keep_prob:keep_prob_tuple[0],
                                                              self.out_keep_prob:keep_prob_tuple[1],
                                                              self.batch_size:len(X_batch),
                                                              self.init_state:next_state })
                else:             
                    _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                                 {self.X:X_batch, self.Y:Y_batch, self.lr:lr,
                                                  self.batch_size:len(X_batch),
                                                  self.in_keep_prob:keep_prob_tuple[0],
                                                  self.out_keep_prob:keep_prob_tuple[1]})
                global_step += 1
                if local_step % 50 == 0:
                    print ('Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f'
                           %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))

            if val_data is not None: # go through testing data, average validation loss and ac 
                val_loss_list, val_acc_list = [], []
                next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
                for X_test_batch, Y_test_batch in zip(self.gen_batch(val_data[0], batch_size),
                                                      self.gen_batch(val_data[1], batch_size)):
                    if (self.stateful) and (len(X_test_batch) == batch_size):
                        v_loss, v_acc, next_state = self.sess.run([self.loss, self.acc, self.final_state],
                                                                  {self.X:X_test_batch, self.Y:Y_test_batch,
                                                                   self.batch_size:len(X_test_batch),
                                                                   self.init_state:next_state,
                                                                   self.in_keep_prob:1.0, self.out_keep_prob:1.0})
                    else:
                        v_loss, v_acc = self.sess.run([self.loss, self.acc], 
                                                      {self.X:X_test_batch, self.Y:Y_test_batch,
                                                       self.batch_size:len(X_test_batch),
                                                       self.in_keep_prob:1.0, self.out_keep_prob:1.0})
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
                       "test_loss: %.4f | test_acc: %.4f |" % (val_loss, val_acc), "lr: %.4f" % (lr) )
        # end "for epoch in range(n_epoch)"

        return log
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
        for X_test_batch in self.gen_batch(X_test, batch_size):
            if (self.stateful) and (len(X_test_batch) == batch_size):
                batch_pred, next_state = self.sess.run([self.logits, self.final_state], 
                                                       {self.X:X_test_batch, self.batch_size:len(X_test_batch),
                                                        self.in_keep_prob:1.0, self.out_keep_prob:1.0,
                                                        self.init_state:next_state})
            else:
                batch_pred = self.sess.run(self.logits, {self.X:X_test_batch, self.batch_size:len(X_test_batch),
                                                         self.in_keep_prob:1.0, self.out_keep_prob:1.0})
            batch_pred_list.append(batch_pred)
        return np.argmax(np.vstack(batch_pred_list), 1)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, len_X, batch_size):
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
    # end method list_avg
# end class