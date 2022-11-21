# EyelashNet: A Dataset and A Baseline Method for Eyelash Matting

Official repository for [EyelashNet: A Dataset and A Baseline Method for Eyelash Matting](http://www.cad.zju.edu.cn/home/jin/siga2021/siga2021.htm) appeared in SIGGRAPH ASIA 2021.

Eyelash matting results on daily-captured images  from [Unsplash](https://unsplash.com/.).
<p align="center">
  <img src="pics/results.jpg" title="Original Image"/>
</p>


## Requirements
#### Packages:
- pytorch>=1.2.0
- torchvision>=0.4.0
- tensorboardX
- numpy
- opencv-python
- toml
- easydict
- pprint
- rmi-pytorch

## Models
| Model Name  |    Training Data  | File Size   |MSE|  SAD  | Grad | Conn |
| :------------- |:------|------------:| :-----|----:|----:|----:|
| [ResNet34_En_nomixup](https://drive.google.com/open?id=1kNj33D7x7tR-5hXOvxO53QeCEC8ih3-A) | ISLVRC 2012 | 166MB |N/A|N/A|N/A|N/A|
| [RenderEyelashNet](https://drive.google.com/drive/folders/1jgVknOmsed9ZjyfsUjLa0vUnXtWBjFOa?usp=share_link) |[Render eyelashes](https://drive.google.com/drive/folders/16Fn20KGOFr7j7qyIjLcCQX6tDbE_LRol?usp=share_link)| 288MB |  - |-|-|-|
| [EyelashNet](https://drive.google.com/drive/folders/1Feeg1e4tJvfBqDovyapCLjDbno9gRtG6?usp=share_link) |[Capture eyelashes](https://drive.google.com/drive/folders/1apS34-DIokrC-9Nx4z0Zl7Ko9r7bL1HJ?usp=share_link)| 288MB |  - |-|-|-|

- **ResNet34_En_nomixup**: Model of the customized ResNet-34 backbone trained on ImageNet. Save to `./pretrain/`. Please refer to  [GCA-Matting](https://github.com/Yaoyi-Li/GCA-Matting) for more details.
- **RenderEyelashNet**: Model trained on the rendered eyelash data. Save to `./checkpoints/RenderEyelashNet/`.
- **EyelashNet**: Model trained on the captured eyelash data. Save to `./checkpoints/EyelashNet/`.

## Train and Evaluate on EyelashNet

### Data Preparation
Download  [BaselineTestDataset](https://drive.google.com/drive/folders/1Hf_NpYcNMENovsNBFLxcfxDORkWq7KDN?usp=share_link), [EyelashNet](https://drive.google.com/drive/folders/1apS34-DIokrC-9Nx4z0Zl7Ko9r7bL1HJ?usp=share_link) , [pupil_bg](https://drive.google.com/drive/folders/1Ls4H71023qVmF5A2on7R0YwZs2oPb3vX?usp=share_link) in `./data/` folder.

Download the [coco dataset](http://images.cocodataset.org/zips/val2014.zip) in ./data/coco_bg/ folder.

### Configuration
TOML files are used as configurations in `./config/`. You can find the definition and options in `./utils/config.py`.

Set `ROOT_PATH = "<path to code folder>/"` in `./root_path.py`.

## Run a demo

```
python eyelash_test.py --config=./config/EyelashNet.toml --checkpoint=checkpoints/EyelashNet/best_model.pth --image-dir=<path to  eyelash image folder>  --output=<path to output folder>
```



### Training
We utilize a desktop PC with single NVIDIA GTX 2080 (8GB memory), Intel Xeon 3.6 GHz CPU, and 16GB RAM to train the network. First, you need to set your training and validation data path in configuration (e.g., ./config/EyelashNet.toml):
```toml
[data]
train_fg = ""
train_alpha = ""
train_bg = ""
test_merged = ""
test_alpha = ""
test_trimap = ""
```
You can train the model by 
```bash
./train.sh
```
### Evaluation
To evaluate the trained model  on BaselineTestDataset, set the path of BaselineTestDataset testing and model name in the configuration file `*.toml`:
```toml
[test]
merged = "./data/BaselineTestDataset/image"
alpha = "./data/BaselineTestDataset/mask"
# this will load ./checkpoint/*/best_model.pth
checkpoint = "best_model" 
```
and run the command:
```bash
./test.sh
```

## Agreement

The code, pretrained models and dataset are available for **non-commercial research purposes** only.

## Acknowledgments

This code borrows heavily from [GCA-Matting](https://github.com/Yaoyi-Li/GCA-Matting). 

```
@inproceedings{li2020natural,
  title={Natural image matting via guided contextual attention},
  author={Li, Yaoyi and Lu, Hongtao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  pages={11450--11457},
  year={2020}
}
```

## Citation

If you find this work, code or dataset useful for your research, please cite:

```
@article{10.1145/3478513.3480540,
author = {Xiao, Qinjie and Zhang, Hanyuan and Zhang, Zhaorui and Wu, Yiqian and Wang, Luyuan and Jin, Xiaogang and Jiang, Xinwei and Yang, Yong-Liang and Shao, Tianjia and Zhou, Kun},
title = {EyelashNet: A Dataset and a Baseline Method for Eyelash Matting},
year = {2021},
issue_date = {December 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {40},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/3478513.3480540},
doi = {10.1145/3478513.3480540},
journal = {ACM Trans. Graph.},
month = {dec},
articleno = {217},
numpages = {17},
keywords = {dataset, deep learning, eyelash matting}
}
```

## Contact

qinjie_xiao@zju.edu.cn

jin@cad.zju.edu.cn

