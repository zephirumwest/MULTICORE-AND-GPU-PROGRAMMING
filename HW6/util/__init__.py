import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def get_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize(224),  # ImageNet  224x224 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='/data/hw6/data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='/data/hw6/data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def train_model(model, trainloader, testloader, epochs=10, lr=0.001, device="cuda", save_path="/data/hw6/pretrained/resnet18_cifar10.pth"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")
    
    torch.save(model.state_dict(), save_path)
    print("Training Complete and Model Saved")

def test_model(model, testloader, device="cuda", save_path="result/accuracy.txt"):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    time = 0.0
    cnt = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            starter.record()
            outputs = model(inputs)
            ender.record()
            torch.cuda.synchronize()
            cnt += 1
            time += starter.elapsed_time(ender) 
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # if cnt >= 100 : break
     
            
    print(f'Inference Time: {time/cnt:.4f} ms')
    print(f'Accuracy on test data: {100 * correct / total:.2f}%')
    with open(save_path, "w") as f:
        f.write(f'{100 * correct / total:.2f}')


def init(): 
    class RestrictedTorch:
        def __getattr__(self, name):
            if name in ["empty", "empty_like", "zeros", "ones", "randn", "rand"] :
                return getattr(torch, name)
            raise AttributeError(f"Cannot use: torch.{name}")

    os.sys.modules["mgp"] = RestrictedTorch()

def mgp_test(model, save_path="result/accuracy.txt"):
    _, testloader = get_dataloader()
    test_model(model, testloader, save_path=save_path)
    
    def compare_files(file1, file2):
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            content1 = f1.read()
            content2 = f2.read()
            
        if float(content1) >= 75.00:
            print("Correct.")
        else:
            print("Not Correct")
    
    compare_files(save_path, "ResNet18/accuracy_baseline.txt")
    
def print_tensor_shape(module, input, output):
    module_name = module.__class__.__name__

    if isinstance(input, tuple):
        in_shape = [inp.shape for inp in input if hasattr(inp, 'shape')]
    else:
        in_shape = input.shape if hasattr(input, 'shape') else None

    if isinstance(output, tuple):
        out_shape = [out.shape for out in output if hasattr(out, 'shape')]
    else:
        out_shape = output.shape if hasattr(output, 'shape') else None

    print(f"[{module_name}] Input shape: {in_shape}  =>  Output shape: {out_shape}")
    

