import numpy as np
import matplotlib.pyplot as plt
import math
import sys 
import pandas as pd
import time

def rmse(y_true, y_pred):
    return math.sqrt(np.average((y_true - y_pred) ** 2, axis=0)[0])

def generate_function_values(x_values,is_above=True):
    e = 0.1 # in order to make sure no point is on the line
    function_above = lambda x: -3*x + 1 + e
    function_below = lambda x: -3*x + 1 - e

    if is_above:
        y_values = [np.random.uniform(low = function_above(x),high = 2) for x in x_values]
    else:
        y_values = [np.random.uniform(low = -2, high = function_below(x)) for x in x_values]

    return y_values

def generate_data_points(size):
    x_values = np.random.uniform(size=size)
    while(len(set(x_values)) != size): # ensure all points are different
        x_values = np.random.uniform(size=size)
    
    x_values_zero = x_values[:int(size/2)]
    y_values_zero = generate_function_values(x_values_zero,False)
    x_values_one = x_values[int(size/2):]
    y_values_one = generate_function_values(x_values_one,True)

    return x_values_zero, y_values_zero, x_values_one, y_values_one
    
def PLA(data):
    weight = np.array([0, 0, 0]) # we have w_0 ,x and y parameter
    
    n_iterations = 0
    while(True):
        np.random.shuffle(data)
        n_iterations += 1
        found = False
        for data_point in data:
            if(math.copysign(1,np.matmul(weight, data_point[0])) != data_point[1]) : # signs are not equal
                weight = weight + data_point[0] * data_point[1]
                found = True
                break
        if not(found):
            break
    print(f'Number of iterations needed {n_iterations}')
    return weight/-weight[2] # make the coefficient of y -1

def MLR(X, t, lamb = 0):
    X = np.array(X)
    t = np.array(t)
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X) + lamb * np.identity(X.shape[1])), X.T), t)

    return w

def find_best_lambda(data):
    lambdas = np.linspace(10, 1000, 200, endpoint=False)  
    
    data = data.sample(frac=1, random_state = 1) # shuffle

    train_rmse = []
    test_rmse = []
    data_size =len(data)
    
    for lamb in lambdas:
        current_train_rmse = 0
        current_test_rmse = 0
        for i in range(5):#5-fold cross val
            val = data[int(i*data_size/5):int((i+1)*data_size/5)]
            train = data.loc[set(data.index) - set(val.index)]
            X = train.iloc[:,:-1]
            X.insert(0,'dummy',1)
            t = train.iloc[:,-1:]
            X = np.array(X)
            t = np.array(t)
            w = MLR(X, t, lamb)
            current_train_rmse += rmse(t, np.matmul(X,w))

            test_x = val.iloc[:,:-1]
            test_x.insert(0,'dummy',1)
            test_t = val.iloc[:,-1:]
            test_x = np.array(test_x)
            test_t = np.array(test_t)
            current_test_rmse += rmse(test_t, np.matmul(test_x,w))
       
        train_rmse.append(current_train_rmse  / 5)
        test_rmse.append(current_test_rmse / 5)
    
    plt.plot(lambdas, train_rmse,'r',label = 'train')
    plt.plot(lambdas, test_rmse,'g', label = 'validation')
    plt.legend(loc="upper right")
    plt.xlabel('lambda')
    plt.ylabel('RMSE')
    plt.show()

if sys.argv[1] == 'part1':
    x = np.random.uniform(size=1000)
    y = -3*x + 1
    plt.plot(x,y,color='g',label = 'seperating function f')
    plt.xlabel('x')
    plt.ylabel('y')
    if sys.argv[2] == 'step1':
        data_size = 50
        figure_name = 'part1_step1.png'
    elif sys.argv[2] == 'step2':
        data_size = 100
        figure_name = 'part1_step2.png'
    elif sys.argv[2] == 'step3':
        data_size = 5000
        figure_name = 'part1_step3.png'
    else:
        sys.exit('Invalid argument for step number')
    x_values_zero, y_values_zero, x_values_one, y_values_one = generate_data_points(data_size)
    plt.plot(x_values_zero, y_values_zero, 'r.')
    plt.plot(x_values_one, y_values_one, 'b.')

    data = [(np.array([1, x_values_zero[i], y_values_zero[i]]).T, -1) for i, _ in enumerate(x_values_zero)]
    data.extend([(np.array([1, x_values_one[i], y_values_one[i]]).T, 1) for i, _ in enumerate(x_values_one)])
    w = PLA(data)
    y_new = x*w[1] + w[0]
    plt.plot(x, y_new, color='purple',label = 'decision boundary')
    plt.legend(loc="upper right")
    plt.savefig(figure_name,dpi=200)

elif sys.argv[1] == 'part2':
    if sys.argv[2] == 'step1':
        start = int(round(time.time() * 1000))
        ds = pd.read_csv('ds1.csv', header=None)
        
        X = ds.iloc[:,:-1]
        X.insert(0,'dummy',1)
        t = ds.iloc[:,-1:]
        X = np.array(X)
        t = np.array(t)
        w = MLR(X, t)
        end = int(round(time.time() * 1000))
        print(f'Time to complete step1: {end-start}ms')
    elif sys.argv[2] == 'step2':
        start = int(round(time.time() * 1000))
        ds = pd.read_csv('ds2.csv', header=None)

        X = ds.iloc[:,:-1]
        X.insert(0,'dummy',1)
        t = ds.iloc[:,-1:]
        X = np.array(X)
        t = np.array(t)
        w = MLR(X, t)
        end = int(round(time.time() * 1000))
        print(f'Time to complete step2: {end-start}ms')
    elif sys.argv[2] == 'step3':
        start = int(round(time.time() * 1000))
        ds = pd.read_csv('ds2.csv', header=None)
        
        X = ds.iloc[:,:-1]
        X.insert(0,'dummy',1)
        t = ds.iloc[:,-1:]
        X = np.array(X)
        t = np.array(t)
        w = MLR(X, t, 400)
        end = int(round(time.time() * 1000))
        print(f'Time to complete step3: {end-start}ms')
            
    else:
        sys.exit('Invalid argument for step number')
else:
    sys.exit('Invalid argument for part number')