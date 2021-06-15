# CDR

ICLR‘2021: Robust Early-learning: Hindering the Memorization of Noisy Labels (PyTorch implementation).  

This is the code for the paper:
[Robust Early-learning: Hindering the Memorization of Noisy Labels](https://openreview.net/forum?id=Eql5b1_hTE4)      
Xiaobo Xia, Tongliang Liu, Bo Han, Chen Gong, Nannan Wang, Zongyuan Ge, Yi Chang.

## Dependencies
We implement our methods by PyTorch on NVIDIA Tesla V100 GPU. The environment is as bellow:
- [Ubuntu 16.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version = 1.2.0
- [CUDA](https://developer.nvidia.com/cuda-downloads), version = 10.0
- [Anaconda3](https://www.anaconda.com/)

### Install requirements.txt
~~~
pip install -r requirements.txt
~~~

## Experiments
We verify the effectiveness of the proposed method on synthetic noisy datasets. In this repository, we provide the used [datasets](https://drive.google.com/open?id=1Tz3W3JVYv2nu-mdM6x33KSnRIY1B7ygQ) (the images and labels have been processed to .npy format). You should put the datasets in the folder “data” when you have downloaded them.       
Here is a training example: 
```bash
python main.py \
    --dataset mnist \
    --noise_type symmetric \
    --noise_rate 0.2 \
    --seed 1
```
If you find this code useful in your research, please cite  
```bash
@inproceedings{xia2021robust,
  title={Robust early-learning: Hindering the memorization of noisy labels},
  author={Xia, Xiaobo and Liu, Tongliang and Han, Bo and Gong, Chen and Wang, Nannan and Ge, Zongyuan and Chang, Yi},
  booktitle={ICLR},
  year={2021}
}
```  
