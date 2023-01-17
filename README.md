# DNN-centerNet

### Compiling Corner Pooling Layers

#### For CPUs
```bash
cd <DNN-centerNet dir>/src/center_net/cpools_/
python setup_cpu.py install --user
```

#### For GPUs
```bash
cd <DNN-centerNet dir>/src/center_net/cpools_/
python setup_gpu.py install --user
```

### Train backbone network

```bash
python train_backbone.py
```

### Train CenterNet

```bash
python train_center_net.py
```

### Test an image

```bash
python test_center_net.py <path_to_image>
```
