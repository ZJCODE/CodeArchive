import tensorflow as tf


class NMF:
    def __init__(self, n_user, n_item, n_hidden=100, lamda=0.001, sess=tf.Session()):
        """
        Parameters:
        -----------
        n_user: int
            Dimensions of user (row of rating)
        n_item: int
            Dimensions of item (column of rating)
        n_hidden: int
            Number of hidden dimensions during factorization
        lamda: float
            Penalty parameter Lambda of the error term
        sess: object
            tf.Session() object
        stateful: boolean
            If true, the final state for each batch will be used as the initial state for the next batch 
        """
        self.n_user = n_user
        self.n_item = n_item
        self.n_hidden = n_hidden
        self.lamda = lamda
        self.sess = sess
        self.build_graph()
    # end constructor

    
    def build_graph(self):
        self.add_input_layer()
        self.add_output_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.R = tf.placeholder(tf.float32, [self.n_user, self.n_item])
        self.U = self.call_W('U', [self.n_user, self.n_hidden])
        self.I = self.call_W('I', [self.n_hidden, self.n_item])
        self.lr = tf.placeholder(tf.float32)
    # end method add_input_layer


    def add_output_layer(self):
        self.R_pred = tf.matmul(self.U, self.I)
    # end method add_output_layer


    def add_backward_path(self):
        R_flatten = tf.reshape(self.R, [-1])
        R_pred_flatten = tf.reshape(self.R_pred, [-1])
        zeros = tf.zeros([self.n_user*self.n_item])
        self.non_zero_indices = tf.where(tf.not_equal(R_flatten, zeros))
        R_non_zero = tf.gather(R_flatten, self.non_zero_indices)
        R_pred_non_zero = tf.gather(R_pred_flatten, self.non_zero_indices)
        cost = tf.reduce_sum(tf.abs(tf.subtract(R_non_zero, R_pred_non_zero)))

        norm_sums = tf.reduce_sum(tf.abs(self.U)) + tf.reduce_sum(tf.abs(self.I))
        regu = tf.multiply(norm_sums, self.lamda)
        
        self.loss = tf.add(cost, regu)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method add_backward_path


    def call_W(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.contrib.layers.variance_scaling_initializer())
    # end method _W
# end class