    # Storing Mnist weights


# TO Fix for new pipeline

from torch_custom import *
import tqdm 

# start from 1
initialization = 1
torch.manual_seed(initialization)
np.random.seed(initialization)


# initialization = 5 seed = 0



name = f"weights{initialization}.csv"
train_acc = f"train_accuracies{initialization}.csv"
test_acc = f"test_accuracies{initialization}.csv"

datapath = r"data/mnist.csv"

"""
MODEL INTITIALIZATION AND SETTING UP THE HYPERPARAMETERS

"""

model = MNIST(784,10).to(device) #change the model definition from the torch_custom file to test on other datasets
train_size = 0.7
datasets = MNISTDataset(datapath,train=True,size=train_size) #change the dataset class from the torch_custom file
test_datasets = MNISTDataset(datapath,train=False,size=train_size)
batch_size = 128
dataloader = DataLoader(dataset=datasets,batch_size=batch_size,shuffle=True)
testloader = DataLoader(dataset=test_datasets,batch_size=batch_size)

num_epochs = 50
learning_rate = 0.5

n_total_steps = len(dataloader)
print(n_total_steps)

#torch.save(model.state_dict(), f"model_{initialization}.pth")

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


w_num = 0
model_keys, model_shapes = model_summary(model)
for shapes in model_shapes:
    w_num += shapes             #shapes[0]*shapes[1]
weights = np.zeros(shape=(n_total_steps*num_epochs,w_num))
print(n_total_steps*num_epochs,w_num)

acc = []
for i in range(0,25):
    r = 45*(i+1)+5
    acc.append(r)


j = 0

"""
TRAINING AND VALIDATION LOOPS 
"""

for epoch in range(num_epochs):

 
    model.train()
    for i,(data,targets) in enumerate(dataloader):
        weights[j] = get_weights(model=model)
        data = data.to(device)
        targets = targets.float()

        targets = targets.to(device)
   
      
    
        outputs = model(data)
        loss = criterion(outputs,targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1)%200==0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
        if epoch*n_total_steps+i in acc:
            get_accuracy(model=model,name =train_acc,dataloader=dataloader,device=device,save_weights=True)
            get_accuracy(model=model,name = test_acc,dataloader=testloader,device=device,save_weights=True)
    print(epoch)
    print("train",get_accuracy(model=model,name =train_acc,device=device,dataloader=dataloader))
    print("test",get_accuracy(model=model,name = test_acc,device=device,dataloader=testloader))


np.savetxt(f"weights{initialization}.csv",weights,delimiter=',')
weights = pd.DataFrame(weights)
weights.to_csv(f"weights{initialization}.csv",header=False,index=False)

