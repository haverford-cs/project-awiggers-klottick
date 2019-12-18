from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from process import *
import numpy as np

def main():
    train_data,test_data, current_data = read_csv(source_file, 0)
    train_data = np.array(train_data)
    #shuffle training data
    np.random.shuffle(train_data)
    #split features and labels
    test_data = np.array(test_data)
    y_train = train_data[:,-1]
    X_train = train_data
    X_train = np.delete(X_train, -1, axis=1)
    y_test = test_data[:,-1]
    X_test = test_data
    X_test = np.delete(X_test, -1, axis=1)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    #normalize data
    mean_pixel = X_train.mean(keepdims=True)
    std_pixel = X_train.std(keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel


    #KNN
    print("KNN")
    for i in range(1, 21):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(X_train, y_train)
        print(str(i) + ": " + str(neigh.score(X_test, y_test)))
    
    #Decision trees
    print("Decision Trees")
    for i in range(0, 15):
        dtree = neigh = DecisionTreeClassifier(random_state = 0, max_depth = i*10+1)
        dtree.fit(X_train, y_train)
        print(str(i * 10) + ": " + str(dtree.score(X_test, y_test)))
        print(np.argmax(dtree.feature_importances_))
        print("")
    print("SVM")
    #svm
    svm = SVC(gamma = "auto")
    svm.fit(X_train, y_train)
    print(svm.score(X_test, y_test))

if __name__ == "__main__" :
    main()
