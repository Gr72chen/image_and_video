import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from collections import Counter
import numpy as np
from train_utils import (
    load_data,
    model_defaults,
    create_model,
    args_to_dict,
    add_dict_to_argparser,
)
import argparse
def create_argparser():
    defaults = dict(
        # train_path="/mnt/nas/课题3：乳腺超声造影视频修正BI-RADS分级/all_path_excel.xlsx",
        train_path="/home/ubuntu/Documents/cgl/CEUS_CLASSIFICATION/dataloader/select4_dataset_frames_excel.xlsx",
        # train_path="/home/ubuntu/Documents/cgl/CEUS_CLASSIFICATION/dataloader/cleaned_excel_1.xlsx",
        # test_path = "/home/ubuntu/Documents/cgl/CEUS_CLASSIFICATION/dataloader/valid.xlsx",
        num_frames = 32,
        select_choice="1",
        lr=1e-4,
        batch_size=4,
        split_size = 0.2,
        seed = 30,
        epochs = 50,
        num_splits = 5,
        log_interval = 1,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        out_dir='./results/'
    )
    defaults.update(model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
# 主函数
# 创建参数解析器
parser = create_argparser()
args = parser.parse_args()

# 加载数据
dataset = load_data(args)
print(dataset[:1])
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 获取标签列表
targets = np.array(dataset.label)
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

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
print(train_loader)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)