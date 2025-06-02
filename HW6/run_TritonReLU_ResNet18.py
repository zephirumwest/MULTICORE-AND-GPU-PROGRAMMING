import torch
import os
import util

util.init()

from TritonReLU_ResNet18 import TritonReLUResNet18

if __name__ == "__main__":
    model = TritonReLUResNet18(num_classes=10)
    pretrained_save_path = "/data/hw6/pretrained/resnet18_cifar10.pth"
    result_save_path = "result/accuracy.txt"
    
    if os.path.exists(pretrained_save_path):
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(pretrained_save_path))
    else:
         assert False, "Pretrained model does not exist"
        
    util.mgp_test(model, save_path=result_save_path)