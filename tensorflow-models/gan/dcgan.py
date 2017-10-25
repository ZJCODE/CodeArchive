import tensorflow as tf


class DCGAN:
    def __init__(self, G_size):
        self.G_size = G_size
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
        self.G_in = tf.placeholder(tf.float32, [None, self.G_size])
        self.X_in = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.train_flag = tf.placeholder(tf.bool)
    # end method input_layer


    def add_Generator(self):
        self.G_out = self.generate(self.G_in)
    # end method add_Generator


    def add_Discriminator(self):
        self.G_logits = self.discriminate(self.G_out)
        self.X_logits = self.discriminate(self.X_in, reuse=True)
        self.G_prob = tf.sigmoid(self.G_logits)
        self.X_prob = tf.sigmoid(self.X_logits)
    # end method add_Discriminator


    def add_backward_path(self):
        ones = tf.ones_like(self.G_logits)
        zeros = tf.zeros_like(self.G_logits)

        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ones, logits=self.G_logits))
        D_loss_X = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ones, logits=self.X_logits))
        D_loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=zeros, logits=self.G_logits))
        self.D_loss = D_loss_X + D_loss_G

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')):
            self.G_train = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.G_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')):
            self.D_train = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.D_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))

        self.mse = tf.reduce_mean(tf.squared_difference(self.G_out, self.X_in))
    # end method add_backward_path


    def generate(self, X):
        # 100 -> (7, 7, 128) ->  (14, 14, 64) -> (28, 28, 1)
        X = tf.layers.dense(X, 7 * 7 * 128, self.lrelu)
        X = tf.reshape(X, [-1, 7, 7, 128])

        Y = tf.layers.conv2d_transpose(X, 64, [5, 5], strides=(2, 2), padding='SAME')
        Y = tf.layers.batch_normalization(Y, training=self.train_flag, momentum=0.9)
        Y = self.lrelu(Y)

        Y = tf.layers.conv2d_transpose(Y, 1, [5, 5], strides=(2, 2), padding='SAME')
        return tf.tanh(Y)
    # end method generate


    def discriminate(self, X, reuse=False):
        # (28, 28, 1) -> (14, 14, 64) -> (7, 7, 128) -> 1
        Y = tf.layers.conv2d(X, 64, [5, 5], strides=(2, 2), padding='SAME', name='conv1', reuse=reuse)
        Y = tf.layers.batch_normalization(Y, training=self.train_flag, name='bn1', reuse=reuse, momentum=0.9)
        Y = self.lrelu(Y)

        Y = tf.layers.conv2d(Y, 128, [5, 5], strides=(2, 2), padding='SAME', name='conv2', reuse=reuse)
        Y = tf.layers.batch_normalization(Y, training=self.train_flag, name='bn2', reuse=reuse, momentum=0.9)
        Y = self.lrelu(Y)

        flat = tf.reshape(Y, [-1, 7 * 7 * 64])
        output = tf.layers.dense(flat, 1, name='out', reuse=reuse)
        return output
    # end method discriminate


    def lrelu(self, X, alpha=0.2):
        return tf.maximum(X, X * alpha)
    # end method lrelu
# end class
