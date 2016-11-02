from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from mnist import MNIST
def load_data():
    """To Load Data from the MNIST dataset
    
    Args:
        None

    Returns:
        Train Data, Test Data
        """
    mndata = MNIST('./data')
    train_data = mndata.load_training()
    test_data = mndata.load_testing()
    return train_data, test_data

def neural_network_1(train_data):
    clf = MLPClassifier(hidden_layer_sizes=(4000,), random_state=1)
    clf.fit(train_data[0], train_data[1]);
    # joblib.dump(clf, 'trained_models/nn1.model')
    return clf

def compute_score(classifier, test_data):
    prediction_labels = classifier.predict(test_data[0])
    correctly_predicted = 0
    for i in xrange(len(test_data[1])):
        if prediction_labels[i] == test_data[1][i]:
            correctly_predicted += 1
    #print 'Score:', correctly_predicted, '/', len(test_data[1])
    return (correctly_predicted * 1.0)/len(test_data[1])

def main():
    train_data, test_data = load_data()
    print 'Training Starts'
    clf = neural_network_1(train_data)
    print 'Training Done'
    print 'Score:', compute_score(clf, test_data)

if __name__ == '__main__':
    main()
