# Benchmarking Mnist 
from torch_custom import *



initialization = 3
masked = True

torch.manual_seed(initialization)
np.random.seed(initialization)




name = f"weights{initialization}_sparse.csv"
train_acc = f"train_accuracies{initialization}_sparse.csv"
test_acc = f"test_accuracies{initialization}_sparse.csv"


datapath = r"data/mnist.csv"

# change the model definition from the torch_custom file to test on other datasets
model = MNIST(784, 10).to(device)
model.load_state_dict(torch.load(f"init/mnist/model_{initialization}.pth", map_location = device))
train_size = 0.7
batch_size = 128
# change the dataset class from the torch_custom file
datasets = MNISTDataset(datapath, train=True, size=train_size)
test_datasets = MNISTDataset(datapath, train=False, size=train_size)
dataloader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=test_datasets, batch_size=batch_size)

num_epochs = 50
learning_rate = 0.5


n_total_steps = len(dataloader)
print(n_total_steps)


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

if masked:
    model_tensors = {}
    mask_list = (np.genfromtxt(f"masks/mask_mnist_{initialization}.csv", delimiter=',')+39200).tolist()
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
    print(len(mask_list))
"""
TRAINING LOOP AND VALIDATION LOOP 
"""

for epoch in range(num_epochs):
    model.train()
    for i, (data, targets) in enumerate(dataloader):
        if masked:
            model.mask(model_tensors)
        data = data.to(device)
        targets = targets.float()
        targets = targets.to(device)

        
        outputs = model(data)
        
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1)%200==0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
    if masked:
        model.mask(model_tensors)
    tacc = get_accuracy(model=model, name=train_acc,device=device, dataloader=dataloader)
    print(epoch)
    print("train", tacc)
    print("test", get_accuracy(model=model, name=test_acc, device=device, dataloader=testloader))
    if (tacc>=92):
        break
print("Train")
print(metrics(model=model,device=device, dataloader=dataloader))
print("Test")
print(metrics(model=model,device=device, dataloader=testloader))

