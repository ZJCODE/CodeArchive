import tensorflow as tf


class MLP_GAN:
    def __init__(self, G_size, X_size, lr_G=1e-4, lr_D=1e-4):
        self.G_size = G_size
        self.X_size = X_size
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        with tf.variable_scope('G'):
            self.add_Generator()
        with tf.variable_scope('D'):
            self.add_Discriminator()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.G_in = tf.placeholder(tf.float32, [None, self.G_size]) # random data
        self.X_in = tf.placeholder(tf.float32, [None, self.X_size]) # real data


    def add_Generator(self):
        G_hidden = tf.layers.dense(self.G_in, 128, tf.nn.relu)
        self.G_out = tf.layers.dense(G_hidden, self.X_size)
    # end method add_Generator


    def add_Discriminator(self):
        D_hidden = tf.layers.dense(self.X_in, 128, tf.nn.relu, name='hidden')
        self.X_true_prob = tf.layers.dense(D_hidden, 1, tf.nn.sigmoid, name='out')
        D_hidden = tf.layers.dense(self.G_out, 128, tf.nn.relu, name='hidden', reuse=True)
        self.G_true_prob = tf.layers.dense(D_hidden, 1, tf.nn.sigmoid, name='out', reuse=True)
    # end method add_Discriminator


    def add_backward_path(self):
        self.G_loss = - tf.reduce_mean(tf.log(self.G_true_prob))
        self.D_loss = - tf.reduce_mean(tf.log(self.X_true_prob) + tf.log(1 - self.G_true_prob))
        self.G_train = tf.train.AdamOptimizer(self.lr_G).minimize(self.G_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))
        self.D_train = tf.train.AdamOptimizer(self.lr_D).minimize(self.D_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))
        self.l2_loss = tf.nn.l2_loss(self.G_out - self.X_in)
    # end method add_backward_path
# end class