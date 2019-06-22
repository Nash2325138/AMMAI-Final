# Epic Kitchen Challenge

## Initialization
```
git submodule update --init --recursive
conda env create -f environment.yml
source activate EpicKitchenPytorch1.0
```

## Download pretrained TSM resnet50
```
# Warning: make sure the downloaded file is under pretrained_model/ (not somewhere e.g. src/pretrained_model/)
# TSM resnet50 pretrained on Kinetics (RGB)
wget -P pretrained_model https://hanlab.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth

# Inception I3D pretrained on Kinetics+Charades (flow)
wget -P pretrained_model --no-check-certificate --content-disposition https://github.com/piergiaj/pytorch-i3d/blob/master/models/flow_charades.pt\?raw\=true

# If you want to run models with SEResNeXt101_32x4d as its backbone
pip install pretrainedmodels (under conda env EpicKitchenPytorch1.0)
```

## Extract pose
1. Extract human bounding boxes first and put the root dir in config
2. Install [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md) under lib/ and copy the complied python openpose to python site-package
3. Run
```
cd src/scripts
python extract_openpose.py -rod ../../yolov3_openpose_json -dc ../configs/data/kitchen_TWGC_new_valid.json -s 0 -e 30000

```
## Training command example
```
CUDA_VISIBLE_DEVICES=1 python train.py --configs configs/train/basic.json configs/loss/verb1_noun0.json configs/arch/inception_i3d.json configs/data/kitchen.json
```

## Inference test set example
```
CUDA_VISIBLE_DEVICES=1 python train.py --resume saved/InceptionV1_I3D/0408_181050/checkpoint-epoch29_1.7006-best.pth --configs configs/data/kitchen_test.json --mode test --save_dir ../outputs/I3D_rgb_verb/
```

## Evaluate scores (can only apply predictions of training set and validation set)
```
python scripts/eval_scores.py -f ../outputs/I3D_rgb_verb/valid_unseen.pkl
```

## Late-fuse predictions of two models by averaging scores
```
python scripts/late_fuse.py -f ../outputs/I2D_rgb_verb/valid_unseen.pkl ../outputs/I3D_flow_verb/valid_unseen.pkl
```

## Weighted action scores
```
# Generate results with weighted action (without -ac will be the original noun_distribution * verb_distribution)
python scripts/joint_prob_for_action.py -n scripts/tsm_rgb_verb0.1+noun1_0517_200122_e19_no_crop_valid_unseen.pkl -v scripts/I3D_verb_flow_0513_151447_e12_valid_unseen.pkl  --out unseen_ac.json -rc ../../EPIC_KITCHENS_2018/annotations/EPIC_train_action_labels.csv -ac

# Calculate action score
python scripts/eval_actions.py -rc ../../EPIC_KITCHENS_2018/annotations/EPIC_train_action_labels.csv -ar seen_ac.json
```
## Conditional on prediction (training and inference examples)
An example of inference noun given verb:
### Training
1. Use trained verb model to inference verb on the *training set* like the section `Inference command example` illustrated, and save results as a pickle file. (e.g. InceptionV1_I3D_e40_verb_trainset.pkl)
2. Add the pickle file path as a argument "prior_verb" in the json config file of data loader (e.g. config/data/kitchen.json)
3. Set argument "condition" as "verb" in the model config file.
3. Start training.
### Inference
1. Use trained verb model to inference verb on the *testing set* and save results as a pickle file (e.g. seen_v.pkl)
2. Add the pickle file path as a argument "prior_verb" in data loader config file (e.g. kitchen_test_seen.json)
3. Start inferncing.

## Create submission file example
```
# You may want to create dummy prediction values first (if you don't have all of the results)
python scripts/create_dummy_prediction.py

# Create seen and unseen prediction json files
python scripts/format_submittion.py -v ../outputs/seen_v.pkl -n ../outputs/dummy_seen.pkl --out ../submissions/seen.json
python scripts/format_submittion.py -v ../outputs/dummy_unseen.pkl -n ../outputs/dummy_unseen.pkl --out ../submissions/unseen.json

# Combine seen and unseen prediction to submit
cd ../submissions
zip -j my-submission.zip seen.json unseen.json
```

## Challenge links
- [Action Recognition](https://competitions.codalab.org/competitions/20115#results)
- [Action Anticipation](https://competitions.codalab.org/competitions/20071#results)
- [Object Detection](https://competitions.codalab.org/competitions/20111#results)

## Referenced papers
- [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf) (I3D)
- [Temporal Bilinear Networks for Video Action Recognition](https://arxiv.org/abs/1811.09974) (TBN)
- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383) (TSM)


## Authors
* Zhe Yu Liu [Nash2325138](https://github.com/Nash2325138)
* Ya-Liang Chang (Allen) [amjltc295](https://github.com/amjltc295)
* bob831009
* hsuanlyh1997
* Ke-Jyun 

