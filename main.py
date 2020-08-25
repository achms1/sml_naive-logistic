import scipy.io as sio
import numpy as np      
import math
data= sio.loadmat('mnist_data.mat')    


# In[12]:


trX= data['trX']
trY= data['trY']
tsX= data['tsX']
tsY= data['tsY']
#print (trX.shape)                                               #Printing the shape of training data trX 
#print (trY.shape)                                               #Printing the shape of training labels trY
#print (tsX.shape)                                               #Printing the shape of testing data tsX
#print (tsY.shape)                                               #Printing the shape of testing data tsY


# In[13]:


trX_m=np.array(trX.mean(1))                                     #Calculating Mean of training data
tsX_m=np.array(tsX.mean(1))
trX_sd=np.array(trX.std(1))                                     #Calcuating Standard Deviation of training data
tsX_sd=np.array(tsX.std(1))
trX_m7=trX_m[:6265]                                             #Extracting Mean of trained data of 7 
trX_m8=trX_m[6265:]                                             #Extracting Mean of trained data of 8
trX_sd7=trX_sd[:6265]                                           #Extracting Standard Deviation of trained data of 7
trX_sd8=trX_sd[6265:]                                           #Extracting Standard Deviation of trained data of 8
trX_m7m=np.array(trX_m7.mean(0))
trX_m7sd=np.array(trX_m7.std(0))
trX_m8m=np.array(trX_m8.mean(0))
trX_m8sd=np.array(trX_m8.std(0))
trX_sd7m=np.array(trX_sd7.mean(0))
trX_sd7sd=np.array(trX_sd7.std(0))
trX_sd8m=np.array(trX_sd8.mean(0))
trX_sd8sd=np.array(trX_sd8.std(0))
#Printing the shapes of mean, Standard Deviation for 7 and 8  
#print (trX_m.shape)
#print (tsX_m.shape)
#print (trX_m7.shape)
#print (trX_m8.shape)
#print (trX_sd.shape)
#print (tsX_sd.shape)
#print (trX_sd7.shape)
#print (trX_sd8.shape)


# In[14]:


#Printing the 2x2 matrix 
#print (trX_m7m)                    #0.11452769775108769
#print (trX_m7sd)                   #0.03063240469648838
#print (trX_m8m)                    #0.15015598189369758
#print (trX_m8sd)                   #0.038632488373958954
#print (trX_sd7m)                   #0.28755656517748474
#print (trX_sd7sd)                  #0.038201083694320306
#print (trX_sd8m)                   #0.3204758364888714
#print (trX_sd8sd)                  #0.039960074370658606


# In[15]:


def gaussian_funct(x,mean,std):                         #Implementing Gaussian function 
    p=1./(math.sqrt(2.*math.pi)*std)*(math.exp(-1*(((x - mean)/std)**2.))/2)
    return p


# In[16]:


#Classifing the final test data as class 7 or as class 8
pofy7 = pofy8 = 0
for i in range(len(tsY[0])):
    #print(tsY[0][i])
    if tsY[0][i]== 0.:
        pofy7 = pofy7 + 1
    else:
        pofy8 = pofy8 + 1        
#print(pofy7)
#print(pofy8)
profy_7=pofy7/len(tsY[0])
profy_8=pofy8/len(tsY[0])
#print(profy_7)                    #0.5134865134865135
#print(profy_8)                    #0.4865134865134865




output=[]
count7 = 0
count8 = 0
for i in range(len(tsX_m)):
    pro7m=gaussian_funct(tsX_m[i],trX_m7m,trX_m7sd)
    pro7sd=gaussian_funct(tsX_sd[i],trX_sd7m,trX_sd7sd)
    pro7msd=pro7m*pro7sd
    pro8m=gaussian_funct(tsX_m[i],trX_m8m,trX_m8sd)    
    pro8sd=gaussian_funct(tsX_sd[i],trX_sd8m,trX_sd8sd)
    pro8msd=pro8m*pro8sd
    profy_7f = profy_7 * pro7msd
    profy_8f = profy_8 * pro8msd
    #print(pro7msd)
    #print(pro8msd)
    #print("/n")
    if profy_7f > profy_8f:
        #print(pro7msd)
        #print(pro8msd)
        output.append(0)
        count7 = count7 + 1
    else:
        output.append(1)
        count8 = count8 + 1
        
#print (count7)              #1069
#print (count8)              #933


# In[18]:


count=0
for i in range(len(output)):
    #print(output[i])
    if(tsY[0][i]==output[i]):
        #print(tsY[0][i], output[i])
        count = count+1
#print(count)               #1373


# In[19]:


pxgivy = count/len(tsY[0])
print("NaiveBayes Classifier Outputs")
print("Overall:", pxgivy*100)            #Final Accuracy of the model predicting either 7 or 8      68.58141858141859



# In[10]:


actual7 = actual8 = count7 = count8 = 0
for i in tsY[0]:
    if (i == 0.0):
        actual7 += 1
    elif (i == 1.0):
        actual8 += 1
for i in zip(output, tsY[0]):
    if(i[0]==i[1]==0.0):
        count7 += 1
    elif (i[0]==i[1]==1.0):
        count8 += 1
pred_acrcy7 = count7/actual7    #Final accuracy of prediction of label 7
pred_acrcy8 = count8/actual8    #Final accuracy of prediction of label 8
print ("7:",pred_acrcy7*100)                  #71.40077821011673
print ("8:",pred_acrcy8*100)                  #65.60574948665298






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


