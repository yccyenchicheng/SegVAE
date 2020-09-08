# SegVAE

| [Project Page](https://yccyenchicheng.github.io/SegVAE/) | [Conference Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520154.pdf) | [ArXiv](https://arxiv.org/abs/2007.08397) | 

Controllable Image Synthesis via SegVAE.  
[Yen-Chi Cheng](https://yccyenchicheng.github.io/), [Hsin-Ying Lee](http://vllab.ucmerced.edu/hylee/), [Min Sun](https://aliensunmin.github.io/), and [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/).    
In European Conference on Computer Vision (ECCV), 2020.

PyTorch implementation for our SegVAE. With the proposed VAE-based framework, we are able to learn how to generate diverse and plausible semantic maps given a label-set. This provides flexible user editing for image synthesis.

## Usage
### Prerequisites
* Ubuntu 18.04 or 16.04
* Python >= 3.6 
* PyTorch >= 1.0
* [tensorboardX](https://github.com/lanpa/tensorboardX) (which requires `tensorflow==1.14.0`)

### Installation
```
git clone https://github.com/yccyenchicheng/SegVAE.git
cd SegVAE
```

## Dataset
We experimented on two datasets: [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) and [HumanParsing](https://github.com/lemondan/HumanParsing-Dataset).
You could download the datasets following their instructions.

Or you can download the dataset we used from this link: https://drive.google.com/drive/folders/1ah26mxO3rFLTMcVbgldEzIj28Poq1lf8?usp=sharing.

Create a folder named `data/segvae/` and put the downloaded `.zip` files under `data/segvae/`, and unzip them:
```
mkdir -p data/segvae
mv ~/Downloads/celebamaskhq.zip data/segvae
mv ~/Downloads/humanparsing.zip data/segvae
cd data/segvae
unzip celebamaskhq.zip
unzip humanparsing.zip
cd ../../
```

## Training
To train the model, run
```
python train.py --batch_size [batch_size] --dataset [humanparsing or celebamaskhq]
```
For example,
```
python train.py --batch_size 16 --dataset celebamaskhq
```
The log files will be written into `logs/segvae_logs`.

Then you could run
```
tensorboard --logdir logs/segvae_logs --port 6006
```
and go to http://127.0.0.1:6006 to see the visualization of training logs in the browser.

## Citation
Please cite our paper if you find the code, or paper useful for your research.

Controllable Image Synthesis via SegVAE  
[Yen-Chi Cheng](https://yccyenchicheng.github.io/), [Hsin-Ying Lee](http://vllab.ucmerced.edu/hylee/), [Min Sun](https://aliensunmin.github.io/), and [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)  
European Conference on Computer Vision (ECCV), 2020

```
@inproceedings{cheng2020segvae,
    title={Controllable Image Synthesis via SegVAE},
    author={Cheng, Yen-Chi and Lee, Hsin-Ying and Sun, Min and Yang, Ming-Hsuan},
    booktitle = {European Conference on Computer Vision},
    year={2020},
}
```

## Acknowledgement
This code borrows heavily from [SPADE](https://github.com/NVlabs/SPADE). We also thank [COCO-GAN](https://github.com/hubert0527/COCO-GAN), [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) for the FID calculation and Spectral Norm implementation.