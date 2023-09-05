# BaSNet-pytorch
### Official Pytorch Implementation of '[Background Suppression Network for Weakly-supervised Temporal Action Localization](https://arxiv.org/abs/1911.09963)' (AAAI 2020 Spotlight)

![BaS-Net architecture](https://user-images.githubusercontent.com/16102333/78222568-69945500-7500-11ea-9468-22b1da6d1d77.png)

> **Background Suppression Network for Weakly-supervised Temporal Action Localization**<br>
> Pilhyeon Lee (Yonsei Univ.), Youngjung Uh (Clova AI, NAVER Corp.), Hyeran Byun (Yonsei Univ.)
>
> Paper: https://arxiv.org/abs/1911.09963
>
> **Abstract:** *Weakly-supervised temporal action localization is a very challenging problem because frame-wise labels are not given in the training stage while the only hint is video-level labels: whether each video contains action frames of interest. Previous methods aggregate frame-level class scores to produce video-level prediction and learn from video-level action labels. This formulation does not fully model the problem in that background frames are forced to be misclassified as action classes to predict video-level labels accurately. In this paper, we design Background Suppression Network (BaS-Net) which introduces an auxiliary class for background and has a two-branch weight-sharing architecture with an asymmetrical training strategy. This enables BaS-Net to suppress activations from background frames to improve localization performance. Extensive experiments demonstrate the effectiveness of BaS-Net and its superiority over the state-of-the-art methods on the most popular benchmarks - THUMOS'14 and ActivityNet.*

## (2020/06/16) Our new model is available now!
### Weakly-supervised Temporal Action Localization by Uncertainty Modeling [[Paper](https://arxiv.org/abs/2006.07006)] [[Code](https://github.com/Pilhyeon/WTAL-Uncertainty-Modeling)]

## Prerequisites
### Recommended Environment
* Python 3.5
* Pytorch 1.0
* Tensorflow 1.15 (for Tensorboard)

### Depencencies
You can set up the environments by using `$ pip3 install -r requirements.txt`.

### Data Preparation
1. Prepare [THUMOS'14](https://www.crcv.ucf.edu/THUMOS14/) dataset.
    - We excluded three test videos (270, 1292, 1496) as previous work did.

2. Extract features with two-stream I3D networks
    - We recommend extracting features using [this repo](https://github.com/piergiaj/pytorch-i3d).
    - For convenience, we provide the features we used. You can find them [here](https://drive.google.com/file/d/19BIRy53w2H5J2Nc_mpAbYPVzElReJswe/view?usp=sharing).
    
3. Place the features inside the `dataset` folder.
    - Please ensure the data structure is as below.
   
~~~~
├── dataset
   └── THUMOS14
       ├── gt.json
       ├── split_train.txt
       ├── split_test.txt
       └── features
           ├── train
               ├── rgb
                   ├── video_validation_0000051.npy
                   ├── video_validation_0000052.npy
                   └── ...
               └── flow
                   ├── video_validation_0000051.npy
                   ├── video_validation_0000052.npy
                   └── ...
           └── test
               ├── rgb
                   ├── video_test_0000004.npy
                   ├── video_test_0000006.npy
                   └── ...
               └── flow
                   ├── video_test_0000004.npy
                   ├── video_test_0000006.npy
                   └── ...
~~~~

## Usage

### Running
You can easily train and evaluate BaS-Net by running the script below.

If you want to try other training options, please refer to `options.py`.

~~~~
$ bash run.sh
~~~~

### Evaulation
The pre-trained model can be found [here](https://drive.google.com/file/d/1W9uVOTEvJAOj99RWRUqrk9NS4ahgSOE6/view?usp=sharing).
You can evaluate the model by running the command below.

~~~~
$ bash run_eval.sh
~~~~

## References
We referenced the repos below for the code.

* [STPN](https://github.com/bellos1203/STPN)
* [ActivityNet](https://github.com/activitynet/ActivityNet)

## Citation
If you find this code useful, please cite our paper.

~~~~
@inproceedings{lee2020BaS-Net,
  title={Background Suppression Network for Weakly-supervised Temporal Action Localization},
  author={Lee, Pilhyeon and Uh, Youngjung and Byun, Hyeran},
  booktitle={The 34th AAAI Conference on Artificial Intelligence},
  pages={11320--11327},
  year={2020}
}
~~~~

## Contact
If you have any question or comment, please contact the first author of the paper - Pilhyeon Lee (lph1114@yonsei.ac.kr).
