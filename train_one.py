import os
import time

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,SubsetRandomSampler,TensorDataset
from sklearn.model_selection import KFold
from torchvision import datasets
import numpy as np
from ceus_easy import Resnet
from ceus_with_bmodel import classific_model
from train_utils import (
    load_data,
    model_defaults,
    create_model,
    args_to_dict,
    add_dict_to_argparser,
)
import argparse

def train_one_epoch(epoch,model,train_loader,opt,loss_fn,args):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx,(inputs,labels) in enumerate(train_loader):
        # ceus,bmodel = inputs
        # ceus,bmodel,labels = ceus.to(args.device),bmodel.to(args.device),labels.to(args.device)
        ceus = inputs
        ceus, labels = ceus.to(args.device), labels.to(
            args.device)
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
    train_acc = correct / total
    return train_loss,train_acc

def test_model(model,test_loader,args):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs,labels in test_loader:
            ceus= inputs
            ceus, labels = ceus.to(args.device), labels.to(
                args.device)
            outputs = model(ceus)
            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    return test_acc

def initialize_weights(m):
    if isinstance(m,nn.Conv3d) or isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# def train(dataset,opt,model,loss_fn,args):
#
#     train_loader = DataLoader(dataset[0],batch_size=args.batch_size,shuffle=True,num_workers=10)
#     print("loader",len(train_loader))
#     test_loader = DataLoader(dataset[1], batch_size=args.batch_size,shuffle=False,num_workers=10)
#
#     for epoch in range(args.epochs):
#         start_time = time.time()
#         train_loss,train_acc = train_one_epoch(epoch,model,train_loader,opt,loss_fn,args)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Trian one epoch time: {elapsed_time:.4f}, Train Accuracy: {train_acc:.4f}")
#         if (epoch+1) % args.log_interval == 0:
#             test_acc = test_model(model,test_loader, args)
#             print(f"Test Accuracy: {test_acc:.4f}")
#
#     print(f"Last Test Accuracy: {test_acc:.4f}")
#     print("All Finished")
def train(dataset,opt,model,loss_fn,args):
    kf = KFold(n_splits=args.num_splits,shuffle=True,random_state=12)
    avg_test_acc = 0.0
    for fold,(train_indices,test_indices) in enumerate(kf.split(dataset)):
        print(f"Fold [{fold+1}/{args.num_splits}]")

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset,batch_size=args.batch_size,sampler=train_sampler,num_workers=16)
        test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler,num_workers=16)
        model.apply(initialize_weights)

        for epoch in range(args.epochs):
            start_time = time.time()
            train_loss,train_acc = train_one_epoch(epoch,model,train_loader,opt,loss_fn,args)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Trian one epoch time: {elapsed_time:.4f}, Train Accuracy: {train_acc:.4f}")
            if (epoch+1) % args.log_interval == 0:
                test_acc = test_model(model,test_loader, args)
                print(f"Fold [{fold+1}/{args.num_splits}], Test Accuracy: {test_acc:.4f}")
        avg_test_acc += test_acc
    avg_test_acc /= (fold + 1)

    print(f"AVG Test Accuracy: {avg_test_acc:.4f}")
    print("All Finished")

def save_model(model, file_path='train_test/ceus_easy_fusion_1.pth'):
    torch.save(model.state_dict(), file_path)
    print(f'Model has been saved to {file_path}')

def create_argparser():
    defaults = dict(
        train_path="/mnt/nas/课题3：乳腺超声造影视频修正BI-RADS分级/all_path_excel.xlsx",
        # test_path = "/home/ubuntu/Documents/cgl/CEUS_CLASSIFICATION/dataloader/valid.xlsx",
        num_frames = 32,
        select_choice="1",
        lr=1e-4,
        batch_size=8,
        split_size = 0.2,
        seed = 42,
        epochs = 80,
        num_splits = 5,
        log_interval = 10,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        out_dir='./results/'
    )
    defaults.update(model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
# 主函数
def main():
    # 创建参数解析器
    parser = create_argparser()
    args = parser.parse_args()

    # 加载数据
    dataset = load_data(args)

    # 初始化模型
    model = Resnet(3, 2, 8, 32, [2, 2, 2, 2], [False, False, False, True], [])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device = args.device)
    # model = create_model(
    #     **args_to_dict(args,model_defaults().keys())
    # ).to(device = args.device)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCEWithLogitsLoss()
    # opt = optim.SGD(model.parameters(),lr=5e-4,momentum=0.5,weight_decay=1e-4)
    opt = optim.Adam(model.parameters(),lr=5e-4,weight_decay=1e-4)

    # 训练模型
    train(dataset,opt,model,loss_fn,args)

    # 保存模型
    save_model(model)

if __name__ == "__main__":
    main()

