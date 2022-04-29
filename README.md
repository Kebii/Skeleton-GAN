# Skeleton-GAN
A GCN-based generative adversarial network for skeleton sequence generation
![image](https://github.com/Kebii/Skeleton-GAN/blob/master/vis/hello.gif)
![image](https://github.com/Kebii/Skeleton-GAN/blob/master/vis/hello2.gif)

# Dependencies
- Python >= 3.6
- PyTorch >= 1.2.0
- pyyaml, tqdm, tensorboardX

# Train
Dataset: NTU-RGB+D 120  
Data preparation can be refered to: https://github.com/kenziyuliu/MS-G3D
```
python train.py --config ./config/train_cfg.yaml
```
# Inference
```
python inference.py --config ./config/test_cfg.yaml --T 300 --B 10
```
