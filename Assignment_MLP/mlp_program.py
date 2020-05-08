#!/usr/bin/env python
# coding: utf-8

# In[1]:
<<<<<<< HEAD


=======
>>>>>>> 132886cf6060abdd02edb1451b37f8ae83b7bc0b
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#reading the dataset from csv file
def read_data(file):
    data = pd.read_csv(file, header=None , index_col=None)
    return data
data = read_data('data.txt')
data.head()


# In[3]:


#shuffling and splitting of the dataset
def split_data(data):
    df = pd.DataFrame(data)
    #shuffle the dataset
    df = df.sample(frac=1)
    #split the dataset
    split = np.random.rand(len(df)) < 0.7
    train = np.asmatrix(df[split], dtype = 'float64')
    test = np.asmatrix(df[~split], dtype = 'float64')
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:,-1]
    return X_train,y_train,X_test,y_test
X_train,y_train,X_test,y_test = split_data(data)


# In[4]:


#initializing params
<<<<<<< HEAD
alpha = 0.002
=======
alpha = 0.7
>>>>>>> 132886cf6060abdd02edb1451b37f8ae83b7bc0b
epoch = 100
W = np.zeros(X_train.shape[1]+1)


# In[5]:


#activation function
def activation(z):
        if z>=0:
            return 1
        else:
            return 0   


# In[6]:


#prediction function
def predict(x):
<<<<<<< HEAD
    z = np.dot(x, W[1:]) + W[0]
    g = activation(z)
    return g
=======
   z = np.dot(x, W[1:]) + W[0]
   g = activation(z)
   return g
>>>>>>> 132886cf6060abdd02edb1451b37f8ae83b7bc0b


# In[7]:


#training to learn the weights
def train(X_train, y_train):
<<<<<<< HEAD
        loss_train = []
        train_acc = []
        epochs = range(1,epoch+1)
        for i in range(epoch):
            correct = 0
            cost = 0 
            for x, y in zip(X_train, y_train):
                prediction = predict(x)
                y = np.array(y)[0][0]
                x = np.array(x)[0]
                error = y - prediction
                actual_value = int(y)
                if actual_value == prediction:
                    correct += 1
                W[1:] += alpha * error * x[0]
                W[0] += alpha * error
                cost += error**2
            cost = cost/2     
            training_accuracy =  correct/float(X_train.shape[0])*100.0  
            loss_train.append(cost)
            train_acc.append(training_accuracy)
            print("epoch:"+str(i)+"  weight:"+str(W)+"  learning rate:"+str(alpha)+"  Training Accuracy:"+str(training_accuracy))
        plt.plot(epochs, train_acc, 'g', label='Training accuracy')        
        plt.xlim(0,epoch)
        plt.title('Training accuracy',fontsize=20)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        plt.plot(epochs, loss_train, 'g', label='Training loss')        
        plt.xlim(0,epoch)
        plt.title('Training loss',fontsize=20)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
=======
       loss_train = []
       epochs = range(1,epoch+1)
       for i in range(epoch):
           correct = 0
           for x, y in zip(X_train, y_train):
               prediction = predict(x)
               y = np.array(y)[0][0]
               x = np.array(x)[0]
               error = y - prediction
               actual_value = int(y)
               if actual_value == prediction:
                 correct += 1
               W[1:] += alpha * error * x[0]
               W[0] += alpha * error
           training_accuracy =  correct/float(X_train.shape[0])*100.0      
           loss_train.append(training_accuracy)
           print("epoch:"+str(i)+"  weight:"+str(W)+"  learning rate:"+str(alpha)+"  Training Accuracy:"+str(training_accuracy))
       plt.plot(epochs, loss_train, 'g', label='Training loss')        
       plt.xlim(0,epoch)
       plt.title('Training loss')
       plt.xlabel('Epochs')
       plt.ylabel('Loss')
       plt.legend()
       plt.show()
>>>>>>> 132886cf6060abdd02edb1451b37f8ae83b7bc0b

train(X_train, y_train)            


# In[8]:


#Testset accuracy, Confusion Matrix and Accuracy metrics
def test(X_test, y_test):
    print("Predictions on test data:")
    correct = 0
    tp,fp,tn,fn = 0,0,0,0
    for x,y in zip(X_test,y_test):
        prediction = predict(x)
        actual_value = int(np.array(y)[0][0])
        print("X: "+str(x)+" prediction: "+str(prediction)+" Actual value:"+str(actual_value))
        if actual_value == prediction:
          correct += 1
        if actual_value == 0 and prediction == 0:
          tp += 1
        if actual_value == 1 and prediction ==1:
          tn += 1
        if actual_value == 0 and prediction ==1:
          fn += 1
        if actual_value == 1 and prediction == 0:
          fp += 1  
    test_accuracy =  correct/float(X_test.shape[0])*100.0
    print("Test Accuracy:"+str(test_accuracy))
    print()
    print("Accuracy metrics:")
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print("Accuracy: "+str(accuracy))
    print("Precision: "+str(precision))
    print("Recall: "+str(recall))
    print()    
    print("Confusion matrix:")
    cm = [[tp,fp],[fn,tn]]
    print(cm)
    print()
    df_cm = pd.DataFrame(cm, range(2), range(2))
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.title('Confusion Matrix', fontsize = 20) 
    plt.xlabel('Predicted', fontsize = 15) 
    plt.ylabel('Actual', fontsize = 15) 
<<<<<<< HEAD

plt.show()
=======
    plt.show()
>>>>>>> 132886cf6060abdd02edb1451b37f8ae83b7bc0b
test(X_test, y_test)

