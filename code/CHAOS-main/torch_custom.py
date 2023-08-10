import numpy as np
import pandas as pd 
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import precision_recall_fscore_support as rpf
from sklearn.metrics import accuracy_score

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
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data,targets in dataloader:
            data = data.to(device)
            targets = targets.numpy()
            outputs = model(data).numpy()[0]
            if (model.num_classes != 1):
                outputs = np.argmax(outputs)
                targets = np.argmax(targets)
            y_true.append(targets.tolist())
            y_pred.append(outputs.tolist())

    acc = accuracy_score(y_true,y_pred)

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
            if (model.num_classes != 1):
                outputs = np.argmax(outputs)
                targets = np.argmax(targets)
            y_true.append(targets.tolist())
            y_pred.append(outputs.tolist())

    scores = rpf(y_true,y_pred, average ='macro')

    return scores


class Model(nn.Module):
    def __init__(self,input_size, nodes,num_classes):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(input_size,nodes)
        self.fc2 = nn.Linear(nodes,num_classes)
        self.sigmoid= nn.Sigmoid()
        if(num_classes==2):
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
            weights = init[initilization]
            self.fc1.weight = torch.nn.parameter.Parameter(torch.from_numpy(weights[0]).float())
            self.fc1.bias = torch.nn.parameter.Parameter(torch.from_numpy(weights[1].reshape(-1)).float())
            self.fc2.weight = torch.nn.parameter.Parameter(torch.from_numpy(weights[2]).float())
            self.fc2.bias = torch.nn.parameter.Parameter(torch.from_numpy(weights[3].reshape(-1)).float())




class Data(Dataset):
    def __init__(self, datapath,train,size):
        self.data = pd.read_csv(datapath)

        arr_data = np.array(self.data,dtype=np.float64)

        
        if train==True:
            arr_data = arr_data[:int(size*len(arr_data))]
        else : 
            arr_data = arr_data[int(size*len(arr_data)):]

        self.arr_data = (arr_data)
        min_max_scaler = preprocessing.StandardScaler()
        self.x = min_max_scaler.fit_transform(self.arr_data[:,:-1])
        self.x = torch.from_numpy(self.x)

        self.x = self.x.float()
        self.y = torch.from_numpy(self.arr_data[:,-1])
        self.y = self.y.float()


       
    def __len__(self):
        return len(self.arr_data)
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
    