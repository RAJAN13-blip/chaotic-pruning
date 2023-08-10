#Mnist profiler

from torch.profiler import profile, record_function, ProfilerActivity
from torch_custom import *
import tqdm 

# start from 1
# initialization = 5 seed = 0
initialization = 1
torch.manual_seed(initialization)
np.random.seed(initialization)
num_epochs = 50


name = f"weights{initialization}.csv"
train_acc = f"train_accuracies{initialization}.csv"
test_acc = f"test_accuracies{initialization}.csv"
datapath = r"mnist.csv"



model = MNIST(784,10).to(device) #change the model definition from the torch_custom file to test on other datasets
model.load_state_dict(torch.load(f"model_{initialization}.pth", map_location = device))
train_size = 0.7
datasets = MNISTDataset(datapath,train=True,size=train_size) #change the dataset class from the torch_custom file
test_datasets = MNISTDataset(datapath,train=False,size=train_size)
batch_size = 128
dataloader = DataLoader(dataset=datasets,batch_size=batch_size,shuffle=True)
testloader = DataLoader(dataset=test_datasets,batch_size=batch_size)


learning_rate = 0.5

n_total_steps = len(dataloader)
print(n_total_steps)


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

mask_list = (np.genfromtxt(f"con{initialization}2.csv", delimiter=',')+39200).tolist()
lim0 = 1
model_tensors = {}
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
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/torch_custom'),
        record_shapes=True,profile_memory=True,
        with_stack=True)


if mask:
    prof.start()
    for epoch in range(num_epochs):

    
        model.train()
        for i,(data,targets) in enumerate(dataloader):
            model.mask(model_tensors)
            data = data.to(device)
            targets = targets.float()

            targets = targets.to(device)
        
            outputs = model(data).reshape(-1)
            loss = criterion(outputs,targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    prof.stop()
else:
    prof.start()
    for epoch in range(num_epochs):

    
        model.train()
        for i,(data,targets) in enumerate(dataloader):
            data = data.to(device)
            targets = targets.float()

            targets = targets.to(device)
        
            outputs = model(data)
            loss = criterion(outputs,targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    prof.stop()
