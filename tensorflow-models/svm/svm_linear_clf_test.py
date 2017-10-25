from svm_linear_clf import LinearSVMClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np


if __name__ == '__main__':
    X, y = make_classification(5000, flip_y=0.5)
    y = np.array([1 if label == 1 else -1 for label in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    Y_train = y_train.reshape(-1, 1)
    Y_test = y_test.reshape(-1, 1)

    clf = LinearSVMClassifier(X_train.shape[1])
    log = clf.fit(X_train, Y_train, batch_size=100, val_data=(X_test, Y_test))
    Y_pred = clf.predict(X_test)
    print("linear svm (tensorflow):", (Y_pred.ravel() == Y_test.ravel()).astype(float).mean())

    clf = SVC(kernel='linear')
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print("linear svm (sklearn):", (y_pred == y_test).mean())
