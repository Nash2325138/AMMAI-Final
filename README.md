# AMMAI Final

## Initialization
```
git submodule update --init --recursive
conda env create -f environment.yml
source activate AMMAI_final
```

## Download pretrained weights
```
```

## Training command example
```
CUDA_VISIBLE_DEVICES=1 python train.py --configs configs/train/basic.json configs/loss/verb1_noun0.json configs/arch/inception_i3d.json configs/data/kitchen.json
```

## Inference test set example
```
CUDA_VISIBLE_DEVICES=1 python train.py --resume saved/InceptionV1_I3D/0408_181050/checkpoint-epoch29_1.7006-best.pth --configs configs/data/kitchen_test.json --mode test --save_dir ../outputs/I3D_rgb_verb/
```

## Referenced papers
- 


## Authors
* Zhe Yu Liu [Nash2325138](https://github.com/Nash2325138)
* Ya-Liang Chang (Allen) [amjltc295](https://github.com/amjltc295)
*
