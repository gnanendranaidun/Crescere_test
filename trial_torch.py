# import torch
# print("CUDA Available:", torch.cuda.is_available())
# print("CUDA Devices:", torch.cuda.device_count())
# print("CUDA Version:", torch.version.cuda)
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Should return False on macOS
