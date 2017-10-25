from rnn_regr import RNNRegressor
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


TIME_STEPS = 20
BATCH_SIZE = 50


class TimeSeriesGen:
    def __init__(self, batch_start, time_steps, batch_size):
        self.batch_start = batch_start
        self.time_steps = time_steps
        self.batch_size = batch_size
        
    def next_batch(self):
        ts = np.arange(self.batch_start, self.batch_start + self.time_steps * self.batch_size)
        ts = ts.reshape([self.batch_size, self.time_steps]) / (10 * np.pi)
        self.batch_start += self.time_steps * self.batch_size
        X = np.sin(ts)
        Y = np.cos(ts)
        return X[:, :, np.newaxis], Y[:, :, np.newaxis], ts


if __name__ == '__main__':
    train_gen = TimeSeriesGen(0, TIME_STEPS, BATCH_SIZE)
    test_gen = TimeSeriesGen(1000, TIME_STEPS, BATCH_SIZE)
    model = RNNRegressor(n_step = TIME_STEPS,
                         n_in = 1,
                         n_out = 1,
                         cell_size = TIME_STEPS)

    model.sess.run(tf.global_variables_initializer())
    train_state = model.sess.run(model.init_state, feed_dict={model.batch_size:BATCH_SIZE})
    test_state = model.sess.run(model.init_state, feed_dict={model.batch_size:BATCH_SIZE})
    sns.set(style='white')
    plt.ion()

    for step in range(200):
        X_train, Y_train, _ = train_gen.next_batch()
        _, train_loss, train_state = model.sess.run([model.train_op, model.loss, model.final_state],
                                                    {model.X:X_train,
                                                     model.Y:Y_train,
                                                     model.init_state:train_state,
                                                     model.batch_size:BATCH_SIZE})

        X_test, Y_test, ts = test_gen.next_batch()
        test_loss, test_state, Y_pred = model.sess.run([model.loss, model.final_state, model.time_seq_out],
                                                       {model.X:X_test,
                                                        model.Y:Y_test,
                                                        model.init_state:test_state,
                                                        model.batch_size:BATCH_SIZE})

        # update plotting
        plt.cla()
        plt.plot(ts.ravel(), Y_test.ravel(), label='actual')
        plt.plot(ts.ravel(), Y_pred.ravel(), color='indianred', label='predicted')
        plt.ylim((-1.2, 1.2))
        plt.xlim((ts.ravel()[0], ts.ravel()[-1]))
        plt.legend(fontsize=15)
        plt.draw()
        plt.pause(0.01)
        print('train loss: %.4f | test loss: %.4f' % (train_loss, test_loss))

    plt.ioff()
    plt.show()
