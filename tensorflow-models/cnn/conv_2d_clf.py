import tensorflow as tf
import numpy as np
import math
import sklearn


class Conv2DClassifier:
    def __init__(self, img_size, img_ch, n_out, kernel_size=(5,5), pool_size=(2,2), padding='valid',
                 sess=tf.Session()):
        """
        Parameters:
        -----------
        img_size: tuple
            (height, width) of the image size
        img_ch: int
            Number of image channel
        kernel_size: tuple
            (height, width) of the 2D convolution window
        pool_size: int
            Size of the max pooling windows (assumed square window)
        n_out: int
            Output dimensions
        sess: object
            tf.Session() object 
        """
        self.img_size = img_size
        self.img_ch = img_ch
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        self.n_out = n_out
        self.sess = sess
        self._cursor = None
        self._img_h = img_size[0]
        self._img_w = img_size[1]
        self._n_filter = img_ch
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_conv(32)
        self.add_pooling()
        self.add_conv(64)
        self.add_pooling()
        self.add_fully_connected(1024)
        self.add_output_layer()   
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.float32, [None, self.img_size[0], self.img_size[1], self.img_ch])
        self.Y = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.train_flag = tf.placeholder(tf.bool)
        self._cursor = self.X
    # end method add_input_layer


    def add_conv(self, out_dim, strides=(1, 1)):
        Y = tf.layers.conv2d(inputs = self._cursor,
                             filters = out_dim,
                             kernel_size = self.kernel_size,
                             strides = strides,
                             padding = self.padding,
                             use_bias = True,
                             activation = tf.nn.relu)
        self._cursor = tf.layers.batch_normalization(Y, training=self.train_flag)
        self._n_filter = out_dim
        if self.padding == 'valid':
            self._img_h = int((self._img_h-self.kernel_size[0]+1) / strides[0])
            self._img_w = int((self._img_w-self.kernel_size[1]+1) / strides[1])
        if self.padding == 'same':
            self._img_h = int(self._img_h / strides[0])
            self._img_w = int(self._img_w / strides[1])
    # end method add_conv_layer


    def add_pooling(self):
        self._cursor = tf.layers.max_pooling2d(inputs = self._cursor,
                                               pool_size = self.pool_size,
                                               strides = self.pool_size,
                                               padding = self.padding)
        self._img_h = int(self._img_h / self.pool_size[0])
        self._img_w = int(self._img_w / self.pool_size[1])
    # end method add_maxpool_layer


    def add_fully_connected(self, out_dim):
        flat = tf.reshape(self._cursor, [-1, self._img_h * self._img_w * self._n_filter])
        fc = tf.layers.dense(flat, out_dim, tf.nn.relu)
        fc = tf.layers.batch_normalization(fc, training=self.train_flag)
        self._cursor = tf.nn.dropout(fc, self.keep_prob)
    # end method add_fully_connected


    def add_output_layer(self):
        self.logits = tf.layers.dense(self._cursor, self.n_out)
    # end method add_output_layer


    def add_backward_path(self):
        self.lr = tf.placeholder(tf.float32)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                  labels=self.Y))
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), self.Y), tf.float32))
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method add_backward_path


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, keep_prob=0.5, en_exp_decay=True,
            en_shuffle=True):
        if val_data is None:
            print("Train %d samples" % len(X))
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch):
            if en_shuffle:
                X, Y = sklearn.utils.shuffle(X, Y)
            
            for local_step, (X_batch, Y_batch) in enumerate(zip(self.gen_batch(X, batch_size),
                                                                self.gen_batch(Y, batch_size))): # batch training
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size) 
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                             {self.X:X_batch, self.Y:Y_batch,
                                              self.lr:lr, self.keep_prob:keep_prob,
                                              self.train_flag:True})
                global_step += 1
                if local_step % 50 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f"
                           %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))

            if val_data is not None: # go through test dara, compute averaged validation loss and acc
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
        # end "for epoch in range(n_epoch):"

        return log
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.logits, {self.X:X_test_batch,
                                                     self.keep_prob:1.0,
                                                     self.train_flag:False})
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


    def call_b(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.01))
    # end method _b
# end class