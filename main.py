import sys
import cv2
import numpy as np
from scipy.misc import imresize
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

def train_nn(train_data, iterations=1000, n_neurons=400, dump=True):
    """To Train the Neural Network
    
    Args:
        Training Data
        Iterations (Default = 1000)
        Number of Neurons in Hidden Layer (Default = 400)

    Returns:
        Trained Neural Network"""
    clf = MLPClassifier(hidden_layer_sizes=(n_neurons,),
                        activation='logistic',
                        random_state=1,
                        max_iter=iterations)
    clf.fit(train_data[0], train_data[1]);
    if dump:
        joblib.dump(clf, 'trained_models/nn_iter' + str(iterations) +
                         '_neurons_' + str(n_neurons) + '.model')
    return clf

def load_trained_nn(iterations=1000, n_neurons=400):
    """To Load an Already Trained Neural Network
    
    Args:
        Number of Iterations
        Number of Neurons
        
    Returns:
        Trained Neural Network or None if not found"""
    clf = joblib.load('trained_models/nn_iter' + str(iterations) +
                       '_neurons_' + str(n_neurons) + '.model')
    return clf

def compute_score(classifier, test_data):
    """To Compute Score of NN Classifier given the Data
    
    Args:
        Classifier
        Test Data
        
    Returns:
        Computed Score"""
    prediction_labels = classifier.predict(test_data[0])
    correctly_predicted = 0
    for i in xrange(len(test_data[1])):
        if prediction_labels[i] == test_data[1][i]:
            correctly_predicted += 1
    #print 'Score:', correctly_predicted, '/', len(test_data[1])
    return (correctly_predicted * 1.0)/len(test_data[1])

def main():
    """The Main Function of Program
    
    Args:
        None
        
    Returns:
        None"""

    train_data, test_data = load_data()
    if len(sys.argv) == 1:
        for n_neurons in range(100, 1001, 100):
            clf = train_nn(train_data=train_data, n_neurons=n_neurons)
            print 'Number of Iterations =', clf.n_iter_
            print 'N Neurons =', n_neurons
            print 'Score:', compute_score(clf, test_data)
    elif len(sys.argv) == 4 and sys.argv[1] == 'example':
        clf = load_trained_nn(n_neurons=sys.argv[2])
        orig_img = test_data[0][int(sys.argv[3])]
        img = np.array(orig_img)
        img = np.reshape(img, (28,28))
        img = img/255.0
        img = imresize(img, (280, 280))
        cv2.imshow('Show', img)
        cv2.waitKey(0)
        print 'Predicted Digit:', clf.predict([orig_img])[0]
    else:
        print 'Error'

if __name__ == '__main__':
    main()
