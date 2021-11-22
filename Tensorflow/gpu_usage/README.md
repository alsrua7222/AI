# GPU_USAGE

```
pip install tensorflow-gpu
```

<https://www.tensorflow.org/install/gpu>

```python
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

위에서 name: "/device:GPU:0"가 뜬다면 성공한 것,   
CPU:0가 뜬다면 tensorflow 삭제하고 다시 해볼 것.   
