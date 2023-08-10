# General Benchmarking

from torch_custom import *
import tqdm 

# start from 1
initialization = 5
torch.manual_seed(initialization)
np.random.seed(initialization)
datapath = r"binary_cancer.csv"
num_epochs = 5
learning_rate = 0.5
mask = False
cutoff = 90

train_acc = f"train_accuracies{initialization}.csv"
test_acc = f"test_accuracies{initialization}.csv"


init = np.load('cancer2.npy',allow_pickle=True)
train_size = 0.8
datasets = Data(datapath,train=True,size=train_size) #change the dataset class from the torch_custom file
test_datasets = Data(datapath,train=False,size=train_size)
m = datasets.size()
n = datasets.classes()

nodes = init[0][0].shape[0]

# name = f"weights{initialization}.csv"


"""
MODEL INTITIALIZATION AND SETTING UP THE HYPERPARAMETERS

"""

model = Model(m,nodes,n).to(device) #change the model definition from the torch_custom file to test on other datasets
model.set_weights(initialization,init)

batch_size = 1
dataloader = DataLoader(dataset=datasets,batch_size=batch_size,shuffle=True)
testloader = DataLoader(dataset=test_datasets,batch_size=batch_size)



n_total_steps = len(dataloader)
print(n_total_steps)

#torch.save(model.state_dict(), f"model_{initialization}.pth")

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


w_num = 0
model_keys, model_shapes = model_summary(model)
for shapes in model_shapes:
    w_num += shapes             #shapes[0]*shapes[1]
#weights = np.zeros(shape=(n_total_steps*num_epochs,w_num))
print(n_total_steps*num_epochs,w_num)

if(mask):
    model_tensors = {}
    mask_list = (np.genfromtxt(f"con{initialization}.csv", delimiter=',')).tolist()
    lim0 = 1
    for name, params in model.named_parameters():
        if name.split(".")[1] == "weight":
            shape_ = params.shape
            lim1 = shape_[0]*shape_[1]
            mask = torch.ones(lim1)
            ml = [int(i-lim0) for i in mask_list if i >= lim0 and i-lim0 < lim1]
            lim0 = lim1
            mask[ml] = 0
            mask = torch.reshape(mask, shape_).to(device)
            model_tensors[f'{name.split(".")[0]}'] = mask



"""
TRAINING AND VALIDATION LOOPS 
"""

for epoch in range(num_epochs):

 
    model.train()
    for i,(data,targets) in enumerate(dataloader):
        if mask:
            model.mask(model_tensors)
        data = data.to(device)
        targets = targets.float()

        targets = targets.to(device)
   
        outputs = model(data).reshape(-1)
        loss = criterion(outputs,targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1)%200==0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
    tacc = get_accuracy(model=model, name=train_acc,device=device, dataloader=dataloader)
    print(epoch)
    print("train", tacc)
    print("test", get_accuracy(model=model, name=test_acc, device=device, dataloader=testloader))
    if (tacc>=cutoff):
        break


print("Train")
metrics(model=model,device=device, dataloader=dataloader)
print("Test")
metrics(mod
el=model,device=device, dataloader=testloader)
