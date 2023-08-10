import pandas as pd
import numpy as np
from sklearn import preprocessing
import os

def onehot(Y_dev,num_classes,zero_indexed=False):

    one_hot_encodings = []
    for class_index in Y_dev:
        one_hot_vector = np.zeros(num_classes)

        if not zero_indexed:
            class_index = class_index - 1

        one_hot_vector[int(class_index)] = 1
        one_hot_encodings.append(one_hot_vector)


    Y_dev = np.array(one_hot_encodings)
    return Y_dev


def prepare_dataset(X_train,Y_train,X_test,Y_test,params,zero_index):
    min_max_scaler = preprocessing.StandardScaler()
    
    X_train = min_max_scaler.fit_transform(X_train.T)
    X_train = X_train.T
    X_train = X_train.reshape(-1,params['num_features'],1)

    X_test = min_max_scaler.fit_transform(X_test.T)
    X_test = X_test.T
    X_test = X_test.reshape(-1,params['num_features'],1)

    Y_train = onehot(Y_train, params['num_classes'],zero_index)
    Y_train = Y_train.reshape(-1,params['num_classes'],1)
    Y_test = onehot(Y_test,params['num_classes'],zero_index)
    Y_test = Y_test.reshape(-1,params['num_classes'],1)

    return X_train,Y_train,X_test,Y_test



def vowel(datapath,params,train_size=0.80):
    
    path = os.path.join(datapath,"vowel.csv")
    data = pd.read_csv(path)
    data = np.array(data)


    np.random.seed(42)  
    np.random.shuffle(data)

    train_index = int(train_size*len(data))

    X_train = data[:train_index,:3]
    X_test = data[train_index:,:3]   
    
    Y_train = data[:train_index,3]
    Y_test = data[train_index:,3]

    X_train,Y_train,X_test,Y_test = prepare_dataset(X_train,Y_train,X_test,Y_test,params,False)

    return X_train,Y_train,X_test,Y_test

def diabetes(datapath,params,train_size=0.70):
    
    path = os.path.join(datapath,"diabetes_run.csv")
    data = pd.read_csv(path)
    data = np.array(data)


    np.random.seed(42)  
    np.random.shuffle(data)

    train_index = int(train_size*len(data))

    X_train = data[:train_index,:8]
    X_test = data[train_index:,:8]   
    
    Y_train = data[:train_index,8]
    Y_test = data[train_index:,8]

    X_train,Y_train,X_test,Y_test = prepare_dataset(X_train,Y_train,X_test,Y_test,params,False)

    return X_train,Y_train,X_test,Y_test




def iris(datapath,train_size=0.80,reduced = False):
    
    
    path = os.path.join(datapath,"iris.csv")
    data = pd.read_csv(path)
    data = np.array(data)
    
    
    np.random.seed(42)  
    np.random.shuffle(data)

    labels = data[:,-1]
    labels = np.array(pd.get_dummies(labels))

    train_index = int(train_size*len(data))

    if reduced:
        X_train = data[:train_index,[1,3,4]]
        X_test = data[train_index:,[1,3,4]]
    else:
        X_train = data[:train_index,1:5]
        X_test = data[train_index:,1:5]

    
    Y_train = labels[:train_index]
    Y_test = labels[train_index:]


    sc = preprocessing.StandardScaler()
    
    X_train = sc.fit_transform(X_train)
    X_train = X_train.T

    X_test = sc.fit_transform(X_test)
    X_test = X_test.T
    
    return X_train,Y_train,X_test,Y_test




def sonar(datapath,params,train_size=0.80):
    path = os.path.join(datapath,"binary_sonar.csv")
    
    data = pd.read_csv(path)
    data = np.array(data)

    m,n = data.shape

    np.random.shuffle(data)

    train_index = int(train_size*len(data))

    X_train = data[:train_index,:-1]
    X_test = data[train_index:,:-1]   
    
    Y_train = data[:train_index,-1]
    Y_test = data[train_index:,-1]

    X_train,Y_train,X_test,Y_test = prepare_dataset(X_train,Y_train,X_test,Y_test,params,True)
    
    return X_train,Y_train,X_test,Y_test








def titanic(datapath,train_size=0.80):

    path = os.path.join(datapath,"titatnic.csv")

    data = pd.read_csv(path)
    data = np.array(data)

    np.random.seed(42)  
    np.random.shuffle(data)


    data.Sex[data.Sex == 'male'] = 1
    data.Sex[data.Sex == 'female'] = 0

    data.Embarked[data.Embarked == 'S'] = 0
    data.Embarked[data.Embarked == 'C'] = 1
    data.Embarked[data.Embarked == 'Q'] = 2

    data.drop(['Name','Ticket','Cabin','PassengerId'],axis=1,inplace = True)


    train_index = int(train_size*len(data))

    X_train = data[:train_index]
    X_test = data[train_index:,:3]   
    
    Y_train = data[:train_index,3]
    Y_test = data[train_index:,3]


    
    sc = preprocessing.StandardScaler()
    
    X_train = sc.fit_transform(X_train)
    X_train = X_train.T

    X_test = sc.fit_transform(X_test)
    X_test = X_test.T
    
    return X_train,Y_train,X_test,Y_test



def liver(datapath,params,train_size = 0.8):
    path = os.path.join(datapath,"Indian Liver Patient.csv")
    data = pd.read_csv(path)
    data.dropna(inplace = True)
    data = np.array(data)
    m,n = data.shape
    
    np.random.shuffle(data)

    train_index = int(train_size*len(data))

    X_train = data[:train_index,:-1]
    X_test = data[train_index:,:-1]   
    
    Y_train = data[:train_index,-1]
    Y_test = data[train_index:,-1]

    X_train,Y_train,X_test,Y_test = prepare_dataset(X_train,Y_train,X_test,Y_test,params,False)

    return X_train,Y_train,X_test,Y_test
