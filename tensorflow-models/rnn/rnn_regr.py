import tensorflow as tf


class RNNRegressor:
    def __init__(self, n_step, n_in, n_out, cell_size, sess=tf.Session()):
        """
        Parameters:
        -----------
        n_step: int
            Number of time steps
        n_in: int
            Input dimensions
        n_out: int
            Output dimensions
        cell_size: int
            Number of units in the rnn cell
        sess: object
            tf.Session() object
        """
        self.n_step = n_step
        self.n_in = n_in
        self.n_out = n_out
        self.cell_size = cell_size
        self.sess = sess
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
        self.X = tf.placeholder(tf.float32, [None, self.n_step, self.n_in])
        self.Y = tf.placeholder(tf.float32, [None, self.n_step, self.n_out])
        self.batch_size = tf.placeholder(tf.int32, [])
        self._cursor = self.X
    # end method add_input_layer


    def add_lstm_cells(self):
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size)
    # end method add_lstm_cells


    def add_dynamic_rnn(self):
        self.init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
        self._cursor, self.final_state = tf.nn.dynamic_rnn(self.cell, self._cursor,
                                                           initial_state=self.init_state,
                                                           time_major=False)
    # end method add_dynamic_rnn


    def add_output_layer(self):
        reshaped = tf.reshape(self._cursor, [-1, self.cell_size])
        self.logits = tf.layers.dense(reshaped, self.n_out)
        self.time_seq_out = tf.reshape(self.logits, [-1, self.n_step, self.n_out])
    # end method add_output_layer


    def add_backward_path(self):
        def flatten(tensor):
            return tf.reshape(tensor, [-1])
        self.loss = tf.reduce_mean(tf.squared_difference(flatten(self.logits), flatten(self.Y)))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    # end method add_backward_path
# end class RNNRegressor