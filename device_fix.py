import torch
import numpy as np
import os

# CPU 配置优化
os.environ['MKL_NUM_THREADS'] = '4'  # 为 Intel MKL 设置线程数
torch.set_num_threads(4)  # 设置 PyTorch 线程数

# 全局设备设置
DEVICE = torch.device('cpu')



def get_device():
    """
    获取当前设备配置
    """
    return DEVICE



def ensure_tensor(data, dtype=torch.float32):
    if torch.is_tensor(data):
        return data.to(DEVICE, dtype=dtype)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=dtype, device=DEVICE)
    else:
        return torch.tensor(data, dtype=dtype, device=DEVICE)




def ensure_numpy(tensor):
    """
    确保数据是numpy数组

    Args:
        tensor: PyTorch张量或numpy数组

    Returns:
        numpy.ndarray: numpy数组
    """
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    return np.array(tensor)


def process_batch(batch):
    """
    处理批量数据，确保所有数据都在正确的设备上

    Args:
        batch: 单个张量、张量列表或字典

    Returns:
        处理后的数据，保持原始结构但确保所有张量都在正确设备上
    """
    if torch.is_tensor(batch):
        return batch.to(DEVICE)
    elif isinstance(batch, (list, tuple)):
        return [process_batch(item) for item in batch]
    elif isinstance(batch, dict):
        return {k: process_batch(v) for k, v in batch.items()}
    return batch


class DeviceAwareModule(torch.nn.Module):
    """
    支持设备感知的基础模块类
    """

    def __init__(self):
        super().__init__()
        self.to(DEVICE)

    def ensure_tensor_on_device(self, x):
        """
        确保输入张量在正确的设备上
        """
        return ensure_tensor(x) if x is not None else None


def move_tensors_to_device(model):
    """
    将模型的所有张量移动到指定设备

    Args:
        model: PyTorch模型
    """
    model.to(DEVICE)


def get_device_info():
    """
    获取当前设备配置信息

    Returns:
        dict: 包含设备配置信息的字典
    """
    return {
        "device": str(DEVICE),
        "num_threads": torch.get_num_threads(),
        "mkl_threads": os.environ.get('MKL_NUM_THREADS'),
    }


# 性能监控装饰器
def device_performance_monitor(func):
    """
    监控函数执行时的设备使用情况的装饰器
    """

    def wrapper(*args, **kwargs):
        try:
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

            if start_time:
                start_time.record()

            result = func(*args, **kwargs)

            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                print(f"Function {func.__name__} execution time: {start_time.elapsed_time(end_time):.2f}ms")

            return result
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


# 示例使用
if __name__ == "__main__":
    # 测试设备配置
    print("Current device configuration:")
    print(get_device_info())

    # 测试张量转换
    test_data = np.random.rand(10, 10)
    tensor = ensure_tensor(test_data)
    print(f"Tensor device: {tensor.device}")


    # 测试性能监控装饰器
    @device_performance_monitor
    def test_function():
        x = torch.randn(1000, 1000, device=DEVICE)
        return torch.mm(x, x)


    test_function()