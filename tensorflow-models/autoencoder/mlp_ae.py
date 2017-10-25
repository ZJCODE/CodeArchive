import tensorflow as tf
import sklearn
import numpy as np
import math


class Autoencoder:
    def __init__(self, n_in, encoder_units, sess=tf.Session()):
        self.sess = sess
        self.n_in = n_in
        self.encoder_units = encoder_units
        self.decoder_units = list(reversed(encoder_units))
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_encoders()
        self.add_decoders()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_in])
    # end method add_input_layer


    def add_encoders(self):
        new_layer = self.X
        forward = [self.n_in] + self.encoder_units
        names = ['layer%s'%i for i in range(len(forward)-1)]
        for i in range(len(names)-1):
            new_layer = self.fc(names[i], new_layer, forward[i], forward[i+1], 'encoder')
            new_layer = tf.nn.relu(new_layer)
        self.encoder_op = self.fc(names[-1], new_layer, forward[-2], forward[-1], 'encoder')
    # end method add_encoders


    def add_decoders(self):
        new_layer = self.encoder_op
        forward = self.decoder_units + [self.n_in]
        names = list(reversed(['layer%s'%i for i in range(len(forward)-1)]))
        for i in range(len(names)-1):
            new_layer = self.fc(names[i], new_layer, forward[i], forward[i+1], 'decoder')
            new_layer = tf.nn.relu(new_layer)
        self.decoder_op = self.fc(names[-1], new_layer, forward[-2], forward[-1], 'decoder')
    # end method add_decoders


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.X, self.decoder_op))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    # end method add_backward_path


    def fc(self, name, X, fan_in, fan_out, mode):
        if mode == 'encoder':
            with tf.variable_scope('weights_tied'):
                W = self.call_W(name+'_w', [fan_in,fan_out])
            b = self.call_b(name+'_'+mode+'_b', shape=[fan_out])
        if mode == 'decoder':
            with tf.variable_scope('weights_tied', reuse=True):
                W = tf.transpose(tf.get_variable(name+'_w'))
            b = self.call_b(name+'_'+mode+'_b', shape=[fan_out])
        Y = tf.nn.bias_add(tf.matmul(X, W), b)
        return Y
    # end method fc


    def call_W(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.contrib.layers.variance_scaling_initializer())
    # end method _W


    def call_b(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.01))
    # end method _b


    def fit(self, X_train, val_data, n_epoch=10, batch_size=128, en_shuffle=True):
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        global_step = 0
        for epoch in range(n_epoch):
            X_train = sklearn.utils.shuffle(X_train)
            for local_step, X_batch in enumerate(self.gen_batch(X_train, batch_size)):
                _, loss = self.sess.run([self.train_op, self.loss], {self.X:X_batch})
                if global_step == 0:
                    print("Initial loss: ", loss)
                if (local_step + 1) % 100 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train loss: %.4f"
                           %(epoch+1, n_epoch, local_step+1, int(len(X_train)/batch_size), loss))
                global_step += 1
            
            val_loss_list = []
            for X_test_batch in self.gen_batch(val_data, batch_size):
                v_loss = self.sess.run(self.loss, {self.X:X_test_batch})
                val_loss_list.append(v_loss)
            val_loss = sum(val_loss_list) / len(val_loss_list)
            print ("Epoch %d/%d | train loss: %.4f | test loss: %.4f" %(epoch+1, n_epoch, loss, v_loss))
    # end method fit_transform


    def transform(self, X_test, batch_size=128):
        res = []
        for X_batch in self.gen_batch(X_test, batch_size):
            res.append(self.sess.run(self.encoder_op, {self.X: X_batch}))
        return np.vstack(res)
    # end method transform


    def predict(self, X_test, batch_size=128):
        res = []
        for X_batch in self.gen_batch(X_test, batch_size):
            res.append(self.sess.run(self.decoder_op, {self.X: X_batch}))
        return np.vstack(res)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch
# end class Autoencoder