import numpy as np
import pandas as pd 
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import precision_recall_fscore_support as rpf
from sklearn.metrics import accuracy_score

from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype


from torch.optim.optimizer import (Optimizer, required, _use_grad_for_differentiable, _default_to_fused_or_foreach,
                        _differentiable_doc, _foreach_doc, _maximize_doc)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MNIST(nn.Module):
    def __init__(self,input_size, num_classes):
        super(MNIST,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,30)
#        self.fc3 = nn.Linear(30,15)
        self.fc4 = nn.Linear(30,num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.num_classes = num_classes

    def forward(self,x):
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
#        x = self.fc3(x)
#        x = self.sigmoid(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x 

    def mask(self,model_tensor): 
        with torch.no_grad():
                self.fc1.weight *= model_tensor['fc1']#*self.fc1.weight
                self.fc2.weight *= model_tensor['fc2']#*self.fc2.weight)
#               self.fc3.weight = torch.nn.parameter.Parameter(model_tensor['fc3']*self.fc3.weight)
                self.fc4.weight *= model_tensor['fc4']#*self.fc4.weight)	


class MNISTDataset(Dataset):
    def __init__(self, datapath,train,size):
        self.data = pd.read_csv(datapath)
        arr_data = np.array(self.data,dtype=np.float64)
        
        if train==True:
            arr_data = arr_data[:int(size*len(arr_data))]
        else : 
            arr_data = arr_data[int(size*len(arr_data)):]
        self.arr_data = (arr_data)
        self.x = torch.from_numpy(self.arr_data[:,1:])
        self.x = self.x.float()
        self.y = torch.from_numpy(self.arr_data[:,0])
        self.y = self.y.float()
       
    def __len__(self):
        return len(self.arr_data)

    def __getitem__(self,index):
        return self.x[index], F.one_hot(self.y[index].long(),10)


def model_summary(model):
    """
    get each layer weights, keys and shapes
    """
    assert isinstance(model,nn.Module)==True

    model_keys = []
    model_shapes = []
    for name, params in model.named_parameters():
        if name.split(".")[1]=="weight":
            model_keys.append(name.split(".")[0])
            model_shapes.append(params.shape[0]*params.shape[1])

    return model_keys,model_shapes



def get_weights(model:nn.Module):
    """
    check whether model is instance of nn.Module
    """
    
    W = []    
    for name, param in model.named_parameters():
        if name.split(".")[1]=="weight":
            index = name.split(".")[0]
            temp_tensor = param.clone()
            temp_weights = temp_tensor.detach().cpu().numpy()
            temp_weights = temp_weights.reshape(-1).tolist()
            W.extend(temp_weights)
            # model_dict[f'{index}'] = np.vstack((model_dict[f'{index}'],temp_weights))

    # return model_dict
    return W




def save_weights(model:nn.Module,name):
    """
    check whether model is instance of nn.Module
    """
    W = []
    for layer, param in model.named_parameters():
        if layer.split(".")[1]=="weight":
            # index = name.split(".")[0]
            temp_tensor = param.clone()
            temp_weights = temp_tensor.detach().cpu().numpy()
            temp_weights = temp_weights.reshape(-1)
            W.extend(temp_weights.tolist())

    with open(name,'a',newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(W)
    file.close()
    

    



def get_accuracy(model:nn.Module, name,dataloader,device,save_weights = False):
    model.eval()
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for data,targets in dataloader:
            data = data.to(device)
            
            targets = targets.to(device)
            outputs = model(data)
            if (model.num_classes == 1):
                predictions = torch.round(outputs)
            else:
                predictions = torch.argmax(outputs,1)
                targets = torch.argmax(targets,1)
            n_samples += dataloader.batch_size
            n_correct += (predictions == targets).sum().tolist()

    acc = 100*n_correct/n_samples
    if save_weights:
        with open(name,'a',newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([acc])
        file.close()
    else:
        return acc

def metrics(model:nn.Module,dataloader,device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data,targets in dataloader:
            data = data.to(device)
            
            targets = targets.numpy()
            outputs = model(data).numpy()[0]
            if (model.num_classes == 1):
                outputs = outputs.round()
            else:
                outputs = np.argmax(outputs)
                targets = np.argmax(targets)

            y_true.append(targets)
            y_pred.append(outputs)
 
    scores = rpf(y_true,y_pred, average ='macro')

    return scores


class Model(nn.Module):
    def __init__(self,input_size, nodes,num_classes):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(input_size,nodes)
        self.fc2 = nn.Linear(nodes,num_classes)
        self.sigmoid= nn.Sigmoid()
        self.num_classes=num_classes
        if(num_classes==1):
            self.output = nn.Sigmoid()
        else:
            self.output = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = self.output(x)
        return x 

    def mask(self,model_tensor): 
        with torch.no_grad():
                self.fc1.weight *= model_tensor['fc1']#*self.fc1.weight
                self.fc2.weight *= model_tensor['fc2']#*self.fc2.weight)

    def set_weights(self,initilization,init):
        with torch.no_grad():
            weights = init[initilization-1]
            self.fc1.weight = torch.nn.parameter.Parameter(torch.from_numpy(weights[0]).float(),requires_grad=True).to(device)
            self.fc1.bias = torch.nn.parameter.Parameter(torch.from_numpy(weights[1].reshape(-1)).float(),requires_grad=True).to(device)
            self.fc2.weight = torch.nn.parameter.Parameter(torch.from_numpy(weights[2]).float(),requires_grad=True).to(device)
            self.fc2.bias = torch.nn.parameter.Parameter(torch.from_numpy(weights[3].reshape(-1)).float(),requires_grad=True).to(device)
   
    def infer(self, x):
        self.eval()    
        with torch.no_grad():
            if type(x) is np.ndarray: # only pass np.ndarray if model is on cpu
                x = torch.from_numpy(x.astype(np.float32))
            return self.forward(x).detach().cpu().numpy()
  
  

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

# def backward()


class Data(Dataset):
    # def __init__(self, datapath,train,size):
        
    #     data = pd.read_csv(datapath,header=None)
        
    #     arr_data = np.array(data,dtype=np.float64)
    #     np.random.seed(42)  
    #     np.random.shuffle(arr_data)
        
    #     self.data = arr_data

    #     if train==True:
    #         arr_data = arr_data[:int(size*len(arr_data))]
    #     else : 
    #         arr_data = arr_data[int(size*len(arr_data)):]
        
        
    #     self.arr_data = (arr_data)
    #     min_max_scaler = preprocessing.StandardScaler()
    #     self.x = min_max_scaler.fit_transform(self.arr_data[:,:-1])
    
    #     self.x = torch.from_numpy(self.x)

    #     self.x = self.x.float()
    #     self.y = torch.from_numpy(self.arr_data[:,-1])
    #     self.y = self.y.float()

    def __init__(self, datapath,train,size,data):
        
        if (data == "titanic" or data == "titanic2"):
            X_train,Y_train,X_test,Y_test = titanic()
        elif (data=="iris"):
            X_train,Y_train,X_test,Y_test = iris(False)
        elif (data=="iris3f"):
            X_train,Y_train,X_test,Y_test = iris(True)
        elif(data == "cancer" or data == "cancer2" or data == "bank_note"):
            X_train,Y_train,X_test,Y_test = def_data(datapath)
        elif (data=="Vowel"):
            X_train,Y_train,X_test,Y_test = vowel()
        else:
            X_train,Y_train,X_test,Y_test = def_data(datapath)

        if train==True:
            arr_data = X_train
            self.x = torch.from_numpy(X_train.T).float()
            self.y = torch.from_numpy(Y_train).float()
        else : 
            self.x = torch.from_numpy(X_test.T).float()
            self.y = torch.from_numpy(Y_test).float()
        
        self.arr_data = [self.x,self.y]
                


    def __len__(self):
        return len(self.x)
    def size(self):
        return self.x.shape[1]

    def __getitem__(self,index):
        if self.classes()==1:
            return self.x[index], self.y[index].long()
        else:
            return self.x[index], F.one_hot(self.y[index].long(),self.classes())

    def classes(self):
        classes = np.unique(self.y).shape[0] 
        if (classes==2):
            return 1
        return classes


class descent(Optimizer):
    """Take a coordinate descent step for a random parameter.

    """
    def __init__(self, parameters, lr=1e-3,foreach = None,differentiable = False):
        defaults = {"lr": lr,"foreach" : foreach,"differentiable":differentiable}
        super().__init__(parameters, defaults)


    def _init_group(self, group, params_with_grad, d_p_list):
        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]

        return has_sparse_grad


    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()


        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list)

            sgd(params_with_grad,d_p_list,lr=group['lr'], has_sparse_grad=False,foreach=group['foreach'])
 

        return loss
    

def sgd(params,d_p_list,*,lr,foreach=None, has_sparse_grad = None):
    r"""Functional API that performs SGD algorithm computation.

   has_sparse_grad See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting

        if not torch.jit.is_scripting():
            _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
        else:
            foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(params,d_p_list,lr=lr,has_sparse_grad=has_sparse_grad)



def _single_tensor_sgd(params,d_p_list,*,lr,has_sparse_grad):

    for i, param in enumerate(params):
        d_p = d_p_list[i] #if not maximize else -d_p_list[i]
        param.add_(d_p, alpha=-lr)


def _multi_tensor_sgd(params,grads,*,lr, has_sparse_grad):

    if len(params) == 0:
        return

    grouped_tensors = _group_tensors_by_device_and_dtype([params, grads], with_indices=True)
    for device_params, device_grads, indices in grouped_tensors.values():
        device_has_sparse_grad = any(grad.is_sparse for grad in device_grads)
        if not device_has_sparse_grad:
            torch._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            # foreach APIs don't support sparse
        
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)



class MSE(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    # def forward(self, input):
    #     """
    #     In the forward pass we receive a Tensor containing the input and return a
    #     Tensor containing the output. You can cache arbitrary Tensors for use in the
    #     backward pass using the save_for_backward method.
    #     """
    #     self.save_for_backward(input)
    #     return input.clamp(min=0)

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

def def_data(datapath):
    data = pd.read_csv(datapath,header=None)
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


def iris(reduced):
    data = pd.read_csv(r"data/iris.csv")
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

def vowel():
    data = pd.read_csv("data/Vowel.csv")
    data = np.array(data)
    m,n = data.shape
    np.random.seed(42)  
    np.random.shuffle(data)

    data_train = data[:741].T

    Y_train = data_train[3]-1
    X_train = data_train[:3]

    data_test = data[741:m].T

    Y_test = data_test[3]-1
    X_test = data_test[:3]
    
    min_max_scaler = preprocessing.MinMaxScaler()
    
    X_train = min_max_scaler.fit_transform(X_train.T)
    X_train = X_train.T

    X_test = min_max_scaler.fit_transform(X_test.T)
    X_test = X_test.T
    
    return X_train,Y_train,X_test,Y_test

