# TPAMI21-Not-All-Samples-are-Trustworthy
This is a Pytorch implementation of the model described in our paper:

>Q. Xu, Z. Yang, Y. Jiang, X. Cao, Q. Huang, and Y. Yao. Not All Samples are Trustworthy: Towards Deep Robust SVP Prediction. TPAMI 2021.

## Dependencies
- PyTorch >= 1.0.0
- torchvision
- numpy
- scipy
- Pillow

## Data
The train/val/test split annotation file is in the `data/` folder for `age`/`lfw`/`shoes` datasets. Please unzip `data/images.zip` before training.

For `age` and `lfw`, the data format is `(file_1, file_2, label, attr_id, pair_id, strength)`; for `shoes`, the data format is `(attr_id, img1_id, img2_id, pair_id, label, strength)`.
- `file_1`/`file_2`: file name of images
- `label`: pairwise comparison label (-1/1)
- `attr_id`: attribute id, starting from 0.
- `pair_id`: pair id given a specific attribute, starting from 0. Note that (i,j,-1) and (i,j,1) are the same pair but different samples for the same attribute.
- `strength`: how many times a label is annotated for a pair given a specific attribute.

For custom dataset, you can write a class that inherits from the base class `PairwiseDataset` in `utils.py` like we do in `LFWDataSet`.

## Model
We provide four implementations of our model: two *sparse-learning-based* (**LS-Deep-with-gamma** and **Logit-Deep-with-gamma**), and two *contraction-loss-based* (**Huber-Deep** and **RLogit-Deep**) described in the paper.

## Train
Here is an example to train the models for `lfw` dataset.
```
python train_gamma_lfw.py --lamda=1.2 -e=30 --loss="l2"
python train_gamma_lfw.py --outer=10 --lamda=0.5 -e=5 --loss="logit" -L=20
python train_wo_gamma_lfw.py -l=1e-3 -e=70 --loss="huber" --lamda=0.1
python train_wo_gamma_lfw.py -l=1e-3 -e=70 --loss="rlogit" --lamda=0.1
```

Hyperparameter:
- `lambda` (lambda1 in our paper): controls the sparsity of `gamma` or the contraction margin.
- `e`: the number of epochs.
- `l`: learning rate.

## Citation
Please cite our paper if you use this code in your own work:

```
@inproceedings{xu2021not,
  title={Not All Samples are Trustworthy: Towards Deep Robust SVP Prediction},
  author={Xu, Qianqian and Yang, Zhiyong and Jiang, Yangbangyan and Cao, Xiaochun and Huang, Qingming and Yao, Yuan},
  booktitle={{IEEE} Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}
```
