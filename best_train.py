import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from collections import Counter
import numpy as np

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root='path_to_your_dataset', transform=transform)

# 获取标签列表
targets = np.array(dataset.targets)
class_counts = Counter(targets)

# 计算每个类别的样本数
def stratified_split(indices, labels, train_size=0.7, val_size=0.1):
    train_indices, temp_indices = train_test_split(indices, stratify=labels, test_size=1-train_size, random_state=42)
    val_size_adjusted = val_size / (1 - train_size)
    val_indices, test_indices = train_test_split(temp_indices, stratify=labels[temp_indices], test_size=1-val_size_adjusted, random_state=42)
    return train_indices, val_indices, test_indices

indices = np.arange(len(dataset))
train_indices, val_indices, test_indices = stratified_split(indices, targets)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


import torch.optim as optim

def train_one_epoch(epoch,model,train_loader,opt,loss_fn,args):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx,(inputs,labels) in enumerate(train_loader):
        # ceus,bmodel = inputs
        # ceus,bmodel,labels = ceus.to(args.device),bmodel.to(args.device),labels.to(args.device)
        ceus = inputs
        ceus, labels = ceus.to(args.device), labels.to(args.device)
        opt.zero_grad()
        outputs = model(ceus)
        loss = loss_fn(outputs,labels)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        _,predicted = torch.max(outputs,1)
        # _, predicted = torch.max(d, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # if (batch_idx+1) % args.log_interval == 0:
        #     print(f"Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(train_loader)}], Train Accuracy: {(correct/total):.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total * 100
    return train_loss,train_acc

def test_model(model,test_loader,loss_fn,args):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        running_loss = 0.0
        for inputs,labels in test_loader:
            ceus= inputs
            ceus, labels = ceus.to(args.device), labels.to(
                args.device)
            outputs = model(ceus)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = running_loss / len(test_loader)
    test_acc = correct / total * 100
    return test_acc,test_loss




model = CustomCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(dataset,opt,model,loss_fn,args):
    best_model_wts = None
    best_val_accuracy = 0.0
    for epoch in range(10):

        train_loss,train_acc = train_one_epoch(epoch, model, train_loader, opt, loss_fn, args)
        val_accuracy,val_loss = test_model(model, val_loader, loss_fn, args)
        print(f'Epoch [{epoch + 1}/args.epochs], Loss: {train_loss}.4f,train Accuracy: {train_acc}.4f%, Validation Accuracy: {val_accuracy}.4f%')
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    test_accuracy,test_loss = test_model(model, test_loader, loss_fn, args)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}.4f%')
