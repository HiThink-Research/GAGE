import torch
import io

def tensor_to_bytes(tensor):
    # 将 Tensor 转换为 bytes
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    bytes_data = buffer.getvalue()
    return bytes_data


def bytes_to_tensor(bytes_data):
    # 将 bytes 数据反解析为 Tensor  
    buffer = io.BytesIO(bytes_data)
    tensor_restored = torch.load(buffer)
    return tensor_restored
    