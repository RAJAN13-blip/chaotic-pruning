import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import csv
import time
from resource import *
from sklearn.metrics import precision_recall_fscore_support as met





lip_lr = False    # use lipshitz learning rate
path = ""    # folder where the weights will be stored


def prepare_data(data):
    ## replace with case switch whatever
    # Lot of data functions redundant fix em

    if (data == "cancer" or data == "cancer2"):
        return cancer()
    elif (data == "titanic" or data == "titanic2"):
        return titanic()
    elif (data == "bank_note"):
        return note()
    elif (data == "vowel"):
        accuracy = accuracy2
        return vowel()
    elif (data=="iris"):
        return iris(False)
    elif (data=="iris3f"):
        return iris(True)





def init_params(neurons):
    W1 = np.random.randn(neurons,9)
    b1 = np.random.randn(neurons,1)
    W2 = np.random.randn(1,neurons)
    b2 = np.random.randn(1,1)
    return W1,b1,W2,b2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.where(x>0,x,0)
    
def relu_derivative(x):
    return np.where(x>0,1)
    
def sigmoid_derivative(x):
    return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def one_hot(Y):
    one_hot_Y = np.eye(6)[Y-1]
        
    return one_hot_Y.T
  
    
def forward_prop(W1,b1,W2,b2,X):
    Z1 = np.dot(W1,X).reshape(-1,1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1).reshape(-1,1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2


def back_prop(Z1,A1,Z2,A2,W2,X,Y):
    X = X.reshape(-1,1)
    m = Y.size
    one_hot_Y = Y#one_hot(Y)
    
    dZ2 = (-2)*np.multiply( one_hot_Y.reshape(-1,1) - A2 ,sigmoid_derivative(Z2).reshape(-1,1))
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2,1)
    dZ1 = W2.T.dot(dZ2)*sigmoid_derivative(Z1)
    dW1 = 1 / m *  dZ1.dot(X.T)
    db1 = 1 / m *  np.sum(dZ1,1)
    return dW1,db1,dW2,db2


def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha):
    W1 -= alpha*dW1
    b1 -= alpha*db1.reshape(-1,1)
    W2 -= alpha*dW2
    b2 -= alpha*db2.reshape(-1,1)
    return W1,b1,W2,b2
    
    
def compute_cost(actual,pred):
    
    J = np.sum(np.square(actual-pred))
    return J
    


def populate_list(W,name,trial,path):
    fname = name+'_'+str(trial)+'.csv' # 'W2_1' |  name = W2 or A2 or acc
    fpath = path+fname
    with open(fpath,'a',newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(W)
    file.close()
    return 


def accuracy(Y_train,X_test,*args):
    correct = 0
    for i in range(X_test.shape[1]):
        
        z1, a1, z2, a2 = forward_prop(args[0], args[1],args[2],args[3],X_test[:,i])
        # y_res = one_hot(Y_train[i]).reshape(-1,1)
        y_res = Y_train[i].reshape(-1,1)
        y_pred = np.floor(a2+0.5)
        if(np.array_equal(y_pred,y_res)):
            correct+=1
    # print(f"correct predictions:{correct}/{X_test.shape[1]}")
    return correct/X_test.shape[1]

def accuracy2(Y_train,X_train,*args):
    correct = 0
    for i in range(X_train.shape[1]):
        
        z1, a1, z2, a2 = forward_prop(args[0], args[1],args[2],args[3],X_train[:,i])
        y_res = one_hot(Y_train[i]).reshape(-1,1)
        y_pred = np.floor(a2+0.5)
        if(np.array_equal(y_pred,y_res)):
            correct+=1
    # print(f"correct predictions:{correct}/{X_test.shape[1]}")
    return correct/X_train.shape[1]

def metrics(Y_train,X_test,*args):
    correct = 0
    y_res= []
    y_pred = []
    for i in range(X_test.shape[1]):
        
        z1, a1, z2, a2 = forward_prop(args[0], args[1],args[2],args[3],X_test[:,i])
        # y_res = one_hot(Y_train[i]).reshape(-1,1)
        if(Y_train.ndim>1): 
            y_res.append(np.argmax(Y_train[i]))
            y_pred.append(np.argmax(a2))
        else:
            y_res.append(Y_train[i])
            y_pred.append(np.floor(a2+0.5)[0])
        
    return met(y_res,y_pred, average ='macro')


def read_weights(trial,list_weights):
    W1 = np.copy(list_weights[trial-1][0])
    b1 = np.copy(list_weights[trial-1][1])
    W2 = np.copy(list_weights[trial-1][2])
    b2 = np.copy(list_weights[trial-1][3])
    return W1,b1,W2,b2

def flatten(l):
    return [item for sublist in l for item in sublist]


def masker(mask_list,trial,l1):
    mask = mask_list[trial-1]
    m1 = [i for i in mask if i<l1]
    m1 = np.array(m1)+1
    m1 = m1.tolist()
    m2 = [i for i in mask if i>=l1]
    m2 = np.array(m2)+1
    m2 = m2.tolist()
    return m1,m2



def run_network(trial,X_train,Y_train,X_test,Y_test,weights,l_rate,num_iterations,mask_list_weight,save_weights=False,benchmark = None):

 ########################  initialization
  
    
    W1, b1, W2, b2 = read_weights(trial,weights)
    
    l1 = weights[0][0].shape[0]*weights[0][0].shape[1]
    
    for mask in mask_list_weight[0]:
        W1.flat[mask-1] = 0

    
    for mask in mask_list_weight[1]:
        W2.flat[mask-1-l1] = 0


    if save_weights:
        populate_list(flatten(W1)+flatten(W2),'pure',trial,path)   
        acc = accuracy(Y_test,X_test,W1, b1,W2,b2)
        train_acc = accuracy(Y_train,X_train,W1,b1,W2,b2)
        populate_list([acc,train_acc],'acc',trial,path)
    ###########################################

    test_accuracy = []
    train_accuracy = []
    

    
    for i in range(num_iterations):
        for j in range(X_train.shape[1]):
            

            z1, a1, z2, a2 = forward_prop(W1, b1,W2,b2,X_train[:,j])
            dW1,db1,dW2,db2 = back_prop(z1,a1,z2,a2,W2,X_train[:,j],Y_train[j])
            W1,b1,W2,b2 = update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,l_rate)

            
            for mask in mask_list_weight[0]:
                W1.flat[mask-1] = 0
            
            for mask in mask_list_weight[1]:
                W2.flat[mask-1-l1] = 0

            if save_weights:
                populate_list(flatten(W1)+flatten(W2),'pure',trial,path)   
                acc = accuracy(Y_test,X_test,W1, b1,W2,b2)
                train_acc = accuracy(Y_train,X_train,W1,b1,W2,b2)
                populate_list([acc,train_acc],'acc',trial,path)
           



        if not save_weights:
            acc = accuracy(Y_test,X_test,W1, b1,W2,b2)
            train_acc = accuracy(Y_train,X_train,W1,b1,W2,b2)
        
            test_accuracy.append(acc)
            train_accuracy.append(train_acc)
            

            if benchmark is not None:
                if (train_acc>=benchmark): #and k == 1):
                    print(i,"\nTrain",train_acc)
                    print("Test",acc)
                    
                    print("train",metrics(Y_train,X_train,W1,b1,W2,b2))
                    print("test",metrics(Y_test,X_test,W1,b1,W2,b2))
                    return W1,b1,W2,b2
                    break

    
    if not save_weights:
        return np.array(train_accuracy),np.array(test_accuracy)
            
        

def time_network(trial,X_train,Y_train,X_test,Y_test,weights,l_rate,num_iterations,mask_list_weight,benchmark = None):

 ########################  initialization
  
    
    W1, b1, W2, b2 = read_weights(trial,weights)
    l1 = weights[0][0].shape[0]*weights[0][0].shape[1]
    
    for mask in mask_list_weight[0]:
        W1.flat[mask-1] = 0    
    for mask in mask_list_weight[1]:
        W2.flat[mask-1-l1] = 0
    
    start_time1 = time.time()


    for i in range(num_iterations):
        for j in range(X_train.shape[1]):
            

            z1, a1, z2, a2 = forward_prop(W1, b1,W2,b2,X_train[:,j])
            dW1,db1,dW2,db2 = back_prop(z1,a1,z2,a2,W2,X_train[:,j],Y_train[j])
            W1,b1,W2,b2 = update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,l_rate)

            
            for mask in mask_list_weight[0]:
                W1.flat[mask-1] = 0
            
            for mask in mask_list_weight[1]:
                W2.flat[mask-1-l1] = 0


        train_acc = accuracy(Y_train,X_train,W1,b1,W2,b2)        
        if (train_acc>=benchmark): #and k == 1):
            t = time.time() - start_time1
            print("--- %s seconds ---" % (t))
            print(i,"\nTrain",train_acc)
            return t
                



def cancer():
    data = pd.read_csv("data/binary_cancer.csv",header=None)
    # data.dropna(inplace = True)
    data = np.array(data)
    m,n = data.shape
    np.random.seed(42)  
    np.random.shuffle(data)

    train_index = int(0.8*len(data))

    X_train = data[:train_index,:-1]
    X_test = data[train_index:,:-1]   
    
    Y_train = data[:train_index,-1]
    Y_test = data[train_index:,-1]
    
    
    min_max_scaler = preprocessing.StandardScaler()
    
    X_train = min_max_scaler.fit_transform(X_train.T)
    

    X_test = min_max_scaler.fit_transform(X_test.T)
     
    
    return X_train,Y_train,X_test,Y_test





def titanic():
    
    train = pd.read_csv("data/titanic_train.csv")
    test = pd.read_csv("data/titanic_test.csv")
    Y_test = pd.read_csv("data/titanic_testy.csv")
    Y_train = train.Survived

    Y_test = np.array(Y_test)[:,1]
    Y_train = np.array(Y_train)


    train.drop('Survived',axis =1,inplace=True)
    train.Sex.replace({'male': 1,'female' : 0},inplace = True )
    train.Embarked.replace({'S': 0,'C' : 1, 'Q':2},inplace = True )    
    train.Embarked.fillna(train.Embarked.mode().iloc[0], inplace=True)
    train.drop(['Name','Ticket','Cabin','PassengerId','Age'],axis=1,inplace = True)

    test.Sex.replace({'male': 1,'female' : 0},inplace = True )
    test.Embarked.replace({'S': 0,'C' : 1, 'Q':2},inplace = True )    
    test.Embarked.fillna(test.Embarked.mode().iloc[0], inplace=True)
    test.Fare.fillna(test.Fare.mode().iloc[0], inplace=True)
    test.drop(['Name','Ticket','Cabin','PassengerId','Age'],axis=1,inplace = True)

    # train.Age.fillna(train.Age.median().iloc[0], inplace=True)
    # test.Age.fillna(test.Age.median().iloc[0], inplace=True)

    X_train = np.array(train)
    X_test = np.array(test)
    
    np.random.seed(42)  

    p = np.random.permutation(len(X_train))
    X_train = X_train[p]
    Y_train = Y_train[p]    

    min_max_scaler = preprocessing.StandardScaler()
    
    X_train = min_max_scaler.fit_transform(X_train.T)
    

    X_test = min_max_scaler.fit_transform(X_test.T)
    
    
    return X_train,Y_train,X_test,Y_test


def note():
    data = pd.read_csv("data/binary_banknote.csv",header=None)
    # data.dropna(inplace = True)
    data = np.array(data)
    m,n = data.shape
    np.random.seed(42)  
    np.random.shuffle(data)

    train_index = int(0.8*len(data))

    X_train = data[:train_index,:-1]
    X_test = data[train_index:,:-1]   
    
    Y_train = data[:train_index,-1]
    Y_test = data[train_index:,-1]
    
    
    min_max_scaler = preprocessing.StandardScaler()
    
    X_train = min_max_scaler.fit_transform(X_train.T)
    

    X_test = min_max_scaler.fit_transform(X_test.T)
     
    
    return X_train,Y_train,X_test,Y_test


def vowel():
    data = pd.read_csv("data/vowel.csv")
    data = np.array(data)
    m,n = data.shape
    np.random.seed(42)  
    np.random.shuffle(data)

    data_train = data[:741].T

    Y_train = data_train[3]
    X_train = data_train[:3]

    data_test = data[741:m].T

    Y_test = data_test[3]
    X_test = data_test[:3]
    
    min_max_scaler = preprocessing.MinMaxScaler()
    
    X_train = min_max_scaler.fit_transform(X_train.T)
    X_train = X_train.T

    X_test = min_max_scaler.fit_transform(X_test.T)
    X_test = X_test.T
    
    return X_train,Y_train,X_test,Y_test


def iris(reduced):
    data = pd.read_csv(r"data/Iris.csv")
    data = np.array(data)
    
    m,n = data.shape
    np.random.seed(42)  
    np.random.shuffle(data)

    labels = data[:,-1]
    labels = np.array(pd.get_dummies(labels))

    
    if reduced:
        X_train = data[:120,[1,3,4]]
        X_test = data[120:,[1,3,4]]
    else:
        X_train = data[:120,1:5]
        X_test = data[120:,1:5]

    Y_train = labels[:120]
    Y_test = labels[120:]

    sc = preprocessing.StandardScaler()
    
    X_train = sc.fit_transform(X_train)
    X_train = X_train.T

    X_test = sc.fit_transform(X_test)
    X_test = X_test.T
    
    return X_train,Y_train,X_test,Y_test
