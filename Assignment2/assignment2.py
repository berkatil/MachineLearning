import numpy as np
import matplotlib.pyplot as plt
import math
import sys 
import pandas as pd
import time
from matplotlib import pyplot as plt

def calculate_acc(w,X,y):
    corrects = 0
    for i, point in enumerate(X):
        sigmoid = 1 / (1 + math.exp(-np.dot(w,X[i])))
        if (sigmoid >=0.5) and (y[i] == 1):
            corrects +=1
        elif (sigmoid < 0.5) and (y[i] == -1):
            corrects +=1

    return corrects / len(y)

def calculate_loss(w, X, y):
    loss = 0

    for i in range(len(X)):
        loss += math.log(1 + math.exp(-y[i]*np.dot(w, X[i])))
    return loss / len(X)

def compute_gradient(w,X,y):
    gradient = 0

    for i in range(len(X)):
        gradient += (y[i] * X[i]) / (1 + math.exp(y[i] * np.dot(w, X[i])))

    return -gradient / len(X)

def logistic_regression(X_train, y_train, step_size, gradient_type = 'batch',num_of_features = 18):
    weight = np.zeros(num_of_features+1)
    previous_loss = math.inf
    n_iterations = 0
    losses = []
    if gradient_type == 'batch':
        while(True):
            n_iterations += 1
            gradient = compute_gradient(weight, X_train, y_train)
            weight = weight - step_size * gradient
            current_loss = calculate_loss(weight,X_train,y_train)
            losses.append(current_loss)
            if (previous_loss - current_loss < 0.00001):
                break
            previous_loss = current_loss
    else :
        while(True):
            data_size = len(y_train)
            n_iterations += 1
            batch_losses = []   
            for i in range(5):#batch size is assumed datasize/5
                batch_data_x = X_train[int(i*data_size/5):int((i+1)*data_size/5)]
                batch_data_y = y_train[int(i*data_size/5):int((i+1)*data_size/5)]
                gradient = compute_gradient(weight, batch_data_x, batch_data_y)
                weight = weight - step_size * gradient
                current_loss = calculate_loss(weight,batch_data_x,batch_data_y)
                batch_losses.append(current_loss)

            current_loss = np.mean(batch_losses)
            losses.append(current_loss)
            if (previous_loss - current_loss < 0.00001):
                break
            previous_loss = current_loss

    return weight, losses, n_iterations

def normalize(train, test):
    for column in train:
        max_value = np.max(train[column])
        min_value = np.min(train[column])
        train[column] = (train[column] - min_value) / (max_value - min_value)
        test[column] = (test[column] - min_value) / (max_value - min_value)
    return train,test

data = pd.read_csv('vehicle.csv')
data = data[(data['Class'] == 'saab') | (data['Class'] == 'van')]
data = data.replace({'Class':{'saab':1, 'van':-1}})
data_size = len(data)
data = data.sample(frac=1,random_state=1)
step_sizes = [0.01, 0.1, 1]

if sys.argv[2] == 'step1':
    for step_size in step_sizes:
        accuracies = []
        for i in range(5):#5-fold cross val
            val = data[int(i*data_size/5):int((i+1)*data_size/5)]
            train = data.loc[set(data.index) - set(val.index)]
            X = train.iloc[:,:-1]
            test_x = val.iloc[:,:-1]
            
            X, test_x = normalize(X,test_x)
            test_x.insert(0,'dummy',1)
            X.insert(0,'dummy',1)
            
            y = train.iloc[:,-1:]
            test_t = val.iloc[:,-1:]
            X = np.array(X)
            y = np.array(y)
            start = int(round(time.time() * 1000))
            weight, losses, n_iterations = logistic_regression(X, y, step_size = step_size)
            end = int(round(time.time() * 1000))
            print(f'Step size {step_size} cross val {i} number of iteration is {n_iterations} and total time is {end-start}ms')
            if i == 0:
                plt.plot(losses)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.savefig(f'FullBatch_fold{i}_stepsize{step_size}.png',dpi=100)  
                plt.clf()
            test_x = np.array(test_x)
            test_t = np.array(test_t)

            accuracies.append(calculate_acc(weight, test_x, test_t))
        
        print(f"Full Batch step size {step_size} average accuracy is {np.mean(accuracies)}")

elif sys.argv[2] == 'step2':
    for step_size in step_sizes:
        accuracies = []
        for i in range(5):#5-fold cross val
            val = data[int(i*data_size/5):int((i+1)*data_size/5)]
            train = data.loc[set(data.index) - set(val.index)]
            X = train.iloc[:,:-1]
            test_x = val.iloc[:,:-1]
            
            X, test_x = normalize(X,test_x)
            test_x.insert(0,'dummy',1)
            X.insert(0,'dummy',1)
            
            y = train.iloc[:,-1:]
            test_t = val.iloc[:,-1:]
            X = np.array(X)
            y = np.array(y)
            start = int(round(time.time() * 1000))
            weight, losses, n_iterations = logistic_regression(X, y, step_size = step_size, gradient_type="minibatch")
            end = int(round(time.time() * 1000))
            print(f'Step size {step_size} cross val {i} number of iteration is {n_iterations} and total time is {end-start}ms')
            if i == 0:
                plt.plot(losses)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.savefig(f'Mini Batch_fold{i}_stepsize{step_size}.png',dpi=100)  
                plt.clf()
            test_x = np.array(test_x)
            test_t = np.array(test_t)

            accuracies.append(calculate_acc(weight, test_x, test_t))
        
        print(f"Mini Batch step size {step_size} average accuracy is {np.mean(accuracies)}")