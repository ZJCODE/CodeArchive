import tensorflow as tf
import numpy as np
import math


class ConvAE:
    def __init__(self, img_size, img_ch, kernel_size=(5,5), sess=tf.Session()):
        self.img_size = img_size
        self.img_ch = img_ch
        self.kernel_size = kernel_size
        self.sess = sess
        self._cursor = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_conv('tied_layer_1', [self.kernel_size[0], self.kernel_size[1], self.img_ch, 32])
        self.add_deconv('tied_layer_1', [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch])
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.X = tf.placeholder(tf.float32, [None, self.img_size[0], self.img_size[1], self.img_ch])
        self._cursor = self.X
    # end method add_input_layer


    def add_conv(self, name, filter_shape, strides=1):
        with tf.variable_scope('weights_tied'):
            W = self.call_W(name+'_W', filter_shape)
        Y = tf.nn.conv2d(self._cursor, W, strides=[1,strides,strides,1], padding='SAME')
        Y = tf.nn.bias_add(Y, self.call_b(name+'_conv_b', [filter_shape[-1]]))
        Y = tf.nn.relu(Y)
        self._cursor = Y
    # end method add_conv


    def add_deconv(self, name, output_shape, strides=1):
        with tf.variable_scope('weights_tied', reuse=True):
            W = tf.get_variable(name+'_W')
        Y = tf.nn.conv2d_transpose(self._cursor, W, output_shape, [1,strides,strides,1], 'SAME')
        Y = tf.nn.bias_add(Y, self.call_b(name+'_deconv_b', [output_shape[-1]]))
        self.decoder_op = Y
    # end method add_deconv


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.X, self.decoder_op))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    # end method add_backward_path


    def call_W(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.contrib.layers.variance_scaling_initializer())
    # end method _W


    def call_b(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.01))
    # end method _b


    def fit(self, X_train, val_data, n_epoch=10, batch_size=128):
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        global_step = 0
        for epoch in range(n_epoch):
            # batch training
            for local_step, X_batch in enumerate(self.gen_batch(X_train, batch_size)):
                _, loss = self.sess.run([self.train_op, self.loss], {self.X:X_batch,
                                                                     self.batch_size:len(X_batch)})
                if global_step == 0:
                    print("Initial loss: ", loss)
                if (local_step + 1) % 100 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train loss: %.4f"
                           %(epoch+1, n_epoch, local_step+1, int(len(X_train)/batch_size), loss))
                global_step += 1
            
            val_loss_list = []
            for X_test_batch in self.gen_batch(val_data, batch_size):
                v_loss = self.sess.run(self.loss, {self.X:X_test_batch, self.batch_size:len(X_test_batch)})
                val_loss_list.append(v_loss)
            val_loss = sum(val_loss_list) / len(val_loss_list)
            print ("Epoch %d/%d | train loss: %.4f | test loss: %.4f" %(epoch+1, n_epoch, loss, v_loss))
    # end method fit_transform


    def predict(self, X_test, batch_size=128):
        res = []
        for X_batch in self.gen_batch(X_test, batch_size):
            res.append(self.sess.run(self.decoder_op, {self.X: X_batch, self.batch_size:len(X_batch)}))
        return np.vstack(res)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch
# end class