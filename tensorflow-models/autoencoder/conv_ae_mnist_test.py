from conv_ae import ConvAE
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = (X_train/255.0).reshape(-1, 28, 28, 1)
    X_test = (X_test/255.0).reshape(-1, 28, 28, 1)

    ae = ConvAE((28, 28), 1)
    ae.fit(X_train, X_test, n_epoch=3)
    X_test_pred = ae.predict(X_test)
    
    plt.imshow(X_test[21].reshape(28, 28))
    plt.show()
    plt.imshow(X_test_pred[21].reshape(28, 28))
    plt.show()
