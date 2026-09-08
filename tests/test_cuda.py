import torch

print("torch version:", torch.__version__)
print("cuda disponible:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("usando cpu")