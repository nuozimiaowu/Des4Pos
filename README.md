# Des4Pos
This repository is the official implementation of Des4Pos 🔥🔥🔥🔥🔥🔥

###  Introduction
  Environment description-based localization in large-scale point-cloud maps constructed with multi-sensor systems holds critical significance for future mobile robotics and large-scale autonomous systems, such as in addressing 'last mile' challenges for sensor-equipped delivery robots. Existing text-to-point-cloud localization methods follow a two-stage paradigm: retrieving candidate submaps from the entire point-cloud map using text queries (coarse stage) and then predicting specific coordinates within these submaps (fine stage). However, current approaches face persistent challenges, including the failure of point-cloud encoders to adequately capture local details and long-range spatial relationships, as well as a substantial modality gap between text and point-cloud representations.
To address these limitations, we propose Des4Pos, a novel two-stage localization framework. During the coarse stage, Des4Pos's point-cloud encoder employs a Multi-scale Fusion Attention Mechanism (MFAM) to enhance local geometric features, followed by a bidirectional LSTM module to amplify global spatial relationships. Simultaneously, the Stepped Text Encoder (STE) incorporates cross-modal prior knowledge from CLIP \cite{radford2021learning} and aligns text and point-cloud features using the prior, thereby bridging modality discrepancies. In the fine stage, we propose a Cascaded Residual Attention (CRA) module to fuse cross-modal features and predict relative position offsets, thereby achieving higher localization precision. \\
\indent Experiments on the KITTI360Pose test set demonstrate that Des4Pos achieves state-of-the-art performance in text-to-point-cloud place recognition. Specifically, it attains a top-1 accuracy of 40\% and a top-10 accuracy of 77\% under a 5-meter radius threshold, surpassing the best existing methods by 7\% and 7\%, respectively.

###  Structure overview
<img width="685" alt="image" src="https://github.com/user-attachments/assets/9e2e2dca-a5b3-4b00-832e-c302c4eddce8" />


###  Experimental performance
![image](https://github.com/user-attachments/assets/0afce509-f72e-4a36-b71f-97bbd01c3e37)


##  Installation
Create a conda environment and install basic dependencies:
```bash
git clone https://github.com/nuozimiaowu/Des4Pos
cd MambaPlace

conda create -n mambaplace python=3.10
conda activate mambaplace

# Install the according versions of torch and torchvision
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# Install required dependencies
CC=/usr/bin/gcc-9 pip install -r requirements.txt
```

## Datasets & Backbone

The KITTI360Pose dataset is used in our implementation.

For training and evaluation, we need cells and poses from Kitti360Pose dataset.
The cells and poses folder can be downlowded from [HERE](https://cvg.cit.tum.de/webshare/g/text2pose/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/)  

In addtion, to successfully implement prototype-based map cloning, we need to know the neighbors of each cell. We use direction folder to store the adjacent cells in different directions. 
The direction folder can be downloaded from [HERE](https://drive.google.com/drive/folders/15nsTfN7oQ2uctghRIWo0UgVmJUURzNUZ?usp=sharing)  

If you want to train the model, you need to download the pretrained object backbone [HERE](https://drive.google.com/file/d/1j2q67tfpVfIbJtC1gOWm7j8zNGhw5J9R/view?usp=drive_link):

The KITTI360Pose and the pretrained object backbone is provided by Text2Pos ([paper](https://arxiv.org/abs/2203.15125), [code](https://github.com/mako443/Text2Pos-CVPR2022))

<!-- ```bash
mkdir checkpoints/k360_30-10_scG_pd10_pc4_spY_all/
wget https://cvg.cit.tum.de/webshare/g/text2pose/pretrained_models/pointnet_acc0.86_lr1_p256.pth
mv pointnet_acc0.86_lr1_p256.pth checkpoints/
``` -->

The final directory structure should be:
```
│Des4Pos/
├──dataloading/
├──datapreparation/
├──data/
│   ├──k360_30-10_scG_pd10_pc4_spY_all/
│       ├──cells/
│           ├──2013_05_28_drive_0000_sync.pkl
│           ├──2013_05_28_drive_0002_sync.pkl
│           ├──...
│       ├──poses/
│           ├──2013_05_28_drive_0000_sync.pkl
│           ├──2013_05_28_drive_0002_sync.pkl
│           ├──...
│       ├──direction/
│           ├──2013_05_28_drive_0000_sync.json
│           ├──2013_05_28_drive_0002_sync.json
│           ├──...
├──checkpoints/
│   ├──pointnet_acc0.86_lr1_p256.pth
├──...
```


## Train
After setting up the dependencies and dataset, our models can be trained using the following commands:

### Train Global Place Recognition (Coarse)

```bash
python -m training.coarse --batch_size 64 --coarse_embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/   \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 20 \
  --learning_rate 0.0005 \
  --lr_scheduler step \
  --lr_step 7 \
  --lr_gamma 0.4 \
  --temperature 0.1 \
  --ranking_loss contrastive \
  --hungging_model t5-large \
  --folder_name PATH_TO_COARSE
```

### Train Fine Localization

```bash
python -m training.fine --batch_size 32 --fine_embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 35 \
  --learning_rate 0.0003 \
  --fixed_embedding \
  --hungging_model t5-large \
  --regressor_cell all \
  --pmc_prob 0.5 \
  --folder_name PATH_TO_FINE
```

## Evaluation

### Evaluation on Val Dataset

```bash
python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --no_pc_augment \
    --no_pc_augment_fine \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
    --path_fine ./checkpoints/{PATH_TO_FINE}/{FINE_MODEL_NAME} 
```

### Evaluation on Test Dataset

```bash
python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --use_test_set \
    --no_pc_augment \
    --no_pc_augment_fine \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
    --path_fine ./checkpoints/{PATH_TO_FINE}/{FINE_MODEL_NAME} 
```
## Acknowledgemengt: 
We borrowed some code from Textpos and Text2Loc, and we would like to thank them for their help!

@InProceedings{xia2024text2loc,
      title={Text2Loc: 3D Point Cloud Localization from Natural Language},
      author={Xia, Yan and Shi, Letian and Ding, Zifeng and Henriques, Jo{\~a}o F and Cremers, Daniel},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2024}
    }
