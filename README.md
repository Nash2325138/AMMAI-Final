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
# not yet 
```

## Inference test set example
```
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/arch/Arcface_IR_SE50.json configs/data/asia_legislator_verification.json configs/loss/classification.json configs/train/basic.json --mode test --save_to ../outputs/baseline_evaluator_another_normalize.npz
```

## Referenced papers
- 


## Authors
* Zhe Yu Liu [Nash2325138](https://github.com/Nash2325138)
* Ya-Liang Chang (Allen) [amjltc295](https://github.com/amjltc295)
*
