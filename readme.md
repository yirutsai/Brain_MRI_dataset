# DLMI HW
## Environment
```
conda create -n <env_name> python=3.8
conda activate <env_name>
pip install -r requirements.txt
```
## Download
```
bash download.sh
```

## Brain MRI
### Training
```
(cd ./Brain_MRI)
python3 train.py
```
* --wandb turn on wandb tracking(need to login your account first)

## Breast UltraSound
### Preprocessing
This will turn "Dataset_BUSI_with_GT" into "data"
```
(cd ./Breast-ultrasound-dataset)
python3 preprocess.py
```
### Training
#### Classification
```
python3 train_classify.py
```

#### Segmentation
```
python train_seg.py  --seed <seed> --batch_size <batch_size> --lr <learning_rate> --n_epochs <n_epochs> --clip_grad <clip_grad_norm> --loss_type <loss_type> --output_dir <output_directory> (--wandb)
```
* seed: random seed fixed
* batch_size: batch size
* lr: learning rate
* n_epochs: number of epochs
* loss_type: default=dice, choose from ["dice","BCE","MSE"]
* output_dir: where the image, mask, pred saved

#### Classify-aid Segmentation
```
python3 train_seg_boost.py --seed <seed> --batch_size <batch_size> --lr <learning_rate> --n_epochs <n_epochs> --clip_grad <clip_grad_norm> --loss_type <loss_type> --output_dir <output_directory> (--wandb)
```
* seed: random seed fixed
* batch_size: batch size
* lr: learning rate
* n_epochs: number of epochs
* loss_type: default=dice, choose from ["dice","BCE","MSE"]
* output_dir: where the image, mask, pred saved

### Just predict(no train)
```
python3 generate.py --batch_size <batch_size> --loss_type <loss_type> --output_dir <output_dir>
```
* batch_size: batch size
* loss_type: default=dice, choose from ["dice","BCE","MSE"]
* output_dir: where the image, mask, pred saved

### Metric(mIOU)
```
python3 miou.py --loss_type <loss_type> --output_dir <output_dir>
```
* loss_type: default=dice, choose from ["dice","BCE","MSE"]
* output_dir: where the image, mask, pred saved
