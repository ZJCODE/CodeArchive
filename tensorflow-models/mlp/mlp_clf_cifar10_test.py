from mlp_clf import MLPClassifier
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()

    X_train = (X_train/255.0).mean(axis=3).reshape(-1, 32*32)
    X_test = (X_test/255.0).mean(axis=3).reshape(-1, 32*32)

    clf = MLPClassifier(32*32, 10)
    log = clf.fit(X_train, y_train, val_data=(X_test, y_test))
    pred = clf.predict(X_test)

    final_acc = (pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)
