import  torch
print(f"当前GPU是否可用:{torch.cuda.is_available()}")
print(f"设备名:{torch.cuda.get_device_name()}")