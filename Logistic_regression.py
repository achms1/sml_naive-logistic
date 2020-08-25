import scipy.io as sio
import numpy as np

def split_dataset(labels, dataset):         #Spitting test data
    images_seven = []
    images_eight = []
    for i,value in enumerate(labels[0]):
        if value == 1.0:
            images_eight.append(dataset[i])
        elif value == 0.0:
            images_seven.append(dataset[i])
    images_seven = np.array(images_seven)
    images_eight = np.array(images_eight)
    return images_seven, images_eight

def split_datalabels(labels):               #Splitting test labels
    images_seven = []
    images_eight = []
    for i,value in enumerate(labels[0]):
        if value == 1.0:
            images_eight.append(labels[0][i])
        elif value == 0.0:
            images_seven.append(labels[0][i])
    images_seven = np.array(images_seven)
    images_eight = np.array(images_eight)
    return images_seven, images_eight

def get_features(dataset,is_lr):        #Features Mean and Standard Deviation extration
    mean_data = np.mean(dataset,axis = 1)
    std_data  = np.std(dataset, axis = 1)
    feature_data = []
    for [m,s] in np.nditer([mean_data,std_data]):
        if is_lr == True:
            feature_data.append([1,m,s])
        else:
            feature_data.append([m,s])
    return np.array(feature_data)

def get_class_probablity(labels, value):    #Obtained class probabilities
    count = 0.0
    for item in np.nditer(labels):
        if item == value:
            count = count + 1.0
    return (count / float(len(labels)))
def compute_accuracy(predicted, actual):     #To predict accuracy
    count = 0
    labelcount = 0
    for pred_label,actual_label in np.nditer([predicted,actual]):
        if pred_label == actual_label:
            count += 1
        labelcount += 1
    return (float(count)/float(labelcount))*100


def logistic_regression(n_iter,learning_rate,features,labels):     #logistic regression training
    weights = np.zeros(features.shape[1])
    for iter in range(n_iter):
        score = 1 / ( 1 + np.exp(-(np.dot(features,weights))))
        prediction = labels - score
        gradient = np.dot(features.T,prediction) / features.shape[0]
        weights += learning_rate*gradient
    return weights

def logistic_regression_test(features,weights):                     #logistic regression test data
    predicted_data = []
    iter = np.dot(features,weights)
    for i in iter:
        if (i) >= 0:
            predicted_data.append(1.0)
        else:
            predicted_data.append(0.0)
    return np.array(predicted_data)

def main():
    # Import data from a .mat file
    Numpyfile = sio.loadmat('mnist_data')

    trX = Numpyfile['trX']
    trY = Numpyfile['trY']
    tsX = Numpyfile['tsX']
    tsY = Numpyfile['tsY']
    
    # Split image data into sets containing only images of '7' and '8'.
    trX7, trX8 = split_dataset(trY,trX)
    tsX7, tsX8 = split_dataset(tsY, tsX)

    # Split image labels into sets containing only labels of '7' and '8'.
    trY7, trY8 = split_datalabels(trY)
    tsY7, tsY8 = split_datalabels(tsY)

     # Train logistic regression models
    trX_features = get_features(trX,True)
    weights = logistic_regression(100000,0.3,trX_features,trY[0])

    # Predict accuracy for class '7' and '8'
    tsX7_features = get_features(tsX7,True)
    tsX8_features = get_features(tsX8,True)
    predicted_labels_7_lr = logistic_regression_test(tsX7_features,weights)
    predicted_labels_8_lr = logistic_regression_test(tsX8_features,weights)

    accuracy_7_lr = compute_accuracy(predicted_labels_7_lr,tsY7)
    accuracy_8_lr = compute_accuracy(predicted_labels_8_lr,tsY8)
    print("Logistic Regression Outputs")
    print ("Overall:",(accuracy_7_lr + accuracy_8_lr )/ 2.0)   #71.78898441182815
    print ("7:",accuracy_7_lr)                           #76.9455252918288
    print ("8:",accuracy_8_lr)                           #66.6324435318275
if __name__ == "__main__":
    main()