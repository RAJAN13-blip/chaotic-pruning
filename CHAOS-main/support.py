# functions for network



def prepare_data():
    
    train = pd.read_csv("titanic_train.csv")
    test = pd.read_csv("titanic_test.csv")
    Y_test = pd.read_csv("titanic_testy.csv")
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





def init_params(neurons):
    W1 = np.random.randn(neurons,6)
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


def back_prop(Z1,A1,Z2,A2,W2,X,Y,func):
    X = X.reshape(-1,1)
    m = Y.size
    one_hot_Y = Y#one_hot(Y)
    if func == 'cross_entropy':
        dZ2 = A2 - one_hot_Y.reshape(-1,1)
    

    elif func == 'mse':
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
    
    
def compute_cost(actual,pred,func):
    if func == 'cross_entropy':
        J = -np.sum(np.multiply(actual,np.log(pred)))
        return (-1)*J
    elif func == 'mse':
        J = np.sum(np.square(actual-pred))
        return J
    else:
        return "check loss func"



# def populate_list(W1,w):
#     norm = W1.reshape(1,-1)
#     norm = norm[0]
#     for i in range(len(norm)):
#         w[i].append(norm[i])
#     return w


def populate_list(W,name,trial,path):
    fname = name+'_'+str(trial)+'.csv' # 'W2_1' |  name = W2 or A2 or acc
    fpath = path+init_words+'\\'+name+'\\'+fname
    with open(fpath,'a',newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(W)
    file.close()
    return 

# def accuracy(Y_train,X_train,*args):
#     correct = 0
#     for i in range(X_train.shape[1]):
        
#         z1, a1, z2, a2 = forward_prop(args[0], args[1],args[2],args[3],X_train[:,i])
#         # y_res = one_hot(Y_train[i]).reshape(-1,1)
        
#         if(np.argmax(a2)+1 == Y_train[i]):
#             correct+=1
#     # print(f"correct predictions:{correct}/{X_test.shape[1]}")
#     return correct/X_train.shape[1]
    
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

def write_csv(df,name,trial,path):
    fname = name+'_'+str(trial)+'.csv' # 'W2_1' |  name = W2 or A2 or acc
    fpath = path+init_words+'\\'+name+'\\'+fname
    df.to_csv(fpath,index=False,header=False)

def read_weights(trial,list_weights):
    W1 = np.copy(list_weights[trial-1][0])
    b1 = np.copy(list_weights[trial-1][1])
    W2 = np.copy(list_weights[trial-1][2])
    b2 = np.copy(list_weights[trial-1][3])
    return W1,b1,W2,b2

def flatten(l):
    return [item for sublist in l for item in sublist]








def run_network(trial,X_train,Y_train,X_test,Y_test,neurons,l_rate,num_iterations,lip_lr,loss,mask_list_weight,save_weights=False):

 ########################  initialization
  
    
    W1, b1, W2, b2 = read_weights(trial,listWeights)
    # W1, b1, W2, b2 = init_params(neurons)
 
    
    if len(mask_list_weight[0])>0:
        for mask in mask_list_weight[0]:
            W1.flat[mask-1] = 0

    if(len(mask_list_weight[1])>0):
        for mask in mask_list_weight[1]:
            W2.flat[mask-1-48] = 0


    if save_weights:
        populate_list(flatten(W1)+flatten(W2),'pure',trial,path)   
        acc = accuracy(Y_test,X_test,W1, b1,W2,b2)
        train_acc = accuracy(Y_train,X_train,W1,b1,W2,b2)
        populate_list([acc,train_acc],'acc',trial,path)
    ###########################################

    test_accuracy = []
    train_accuracy = []
    # k = 0
######################### without peturbation
    # print("Without perturbation")
    for i in range(num_iterations):
        for j in range(X_train.shape[1]):
            
            if lip_lr:
                l_rate = 2/(np.linalg.norm(X_train[:,j]))

            z1, a1, z2, a2 = forward_prop(W1, b1,W2,b2,X_train[:,j])
            dW1,db1,dW2,db2 = back_prop(z1,a1,z2,a2,W2,X_train[:,j],Y_train[j],loss)
            W1,b1,W2,b2 = update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,l_rate)

            if len(mask_list_weight[0])>0:
                for mask in mask_list_weight[0]:
                    W1.flat[mask-1] = 0

            if len(mask_list_weight[1])>0:
                for mask in mask_list_weight[1]:
                    W2.flat[mask-1-48] = 0

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
            
            # print(i,"\nacc",acc)
            # print(i,"\nacc train: ",train_acc)
            # print()

            
            # if (i==15):
            if (train_acc>=0.74): #and k == 1):
                print(i,"\nTest",acc)
                print("Train",train_acc)
                break

        
    
    if not save_weights:
        return np.array(train_accuracy),np.array(test_accuracy)
            
        

             


