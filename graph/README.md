
## Prerequisites
- torch_geometric


Usage (example):
```
$ python deepinfomax.py --DS DATASET_NAME --lr 0.001 --num-gc-layers 3
```

To run the batch of experiments on all 8 datasets from the paper, first download the data, then use:

```
$ bash launch.sh
```
Datasets can be downloaded at https://chrsmrrs.github.io/datasets/

## Acknowledgements

This code is a modification of the official InfoGraph code by [fanyun-sun
/
InfoGraph](https://github.com/fanyun-sun/InfoGraph). All code is the same as from the official InfoGraph code except the simple modification for hard sampling is the 
```
measure == 'JSD_hard' 
```

option in [cortex_DIM/functions/gan_losses.py](https://github.com/joshr17/HCL/blob/main/graph/cortex_DIM/functions/gan_losses.py).