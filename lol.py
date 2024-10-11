import cupy
from cuml.common.device_selection import set_global_device_type, get_global_device_type

print(cupy.cuda.runtime.getDeviceProperties(0))