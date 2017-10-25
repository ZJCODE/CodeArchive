from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from logistic import Logistic


if __name__ == '__main__':
    X, y = make_classification(5000, flip_y=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    clf = Logistic(X.shape[1], 2)
    clf.fit(X_train, y_train, val_data=(X_test, y_test))
    y_pred = clf.predict(X_test)
    final_acc = (y_pred == y_test).mean()
    print("logistic (tensorflow): %.4f" % final_acc)

    clf = SVC(kernel='linear')
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print("logistic (sklearn):", (y_pred == y_test).mean())
