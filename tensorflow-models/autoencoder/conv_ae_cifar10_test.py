from conv_ae import ConvAE
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()
    X_train = (X_train/255.0).reshape(-1, 32, 32, 3)
    X_test = (X_test/255.0).reshape(-1, 32, 32, 3)

    ae = ConvAE((32, 32), 3)
    ae.fit(X_train, X_test, n_epoch=5)
    X_test_pred = ae.predict(X_test)
    
    print("Plotting...")
    plt.imshow(X_test[21].reshape(32,32,3))
    plt.show()
    plt.imshow(X_test_pred[21].reshape(32,32,3))
    plt.show()
