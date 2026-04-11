import torch
import torchvision

print("torch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(0))

x = torch.rand(3, 3)
print("before:", x.device)

if torch.cuda.is_available():
    x = x.to("cuda")
    print("after:", x.device)