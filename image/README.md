
## Prerequisites
- Python 3.7 
- PyTorch 1.3.1
- PIL
- OpenCV

## Contrastive Representation Learning
We can train standard (uniform negative sampling) or hard negatives version (with M=1) of [SimCLR](https://arxiv.org/abs/2002.05709) with `main.py`.

flags:
  - `--beta`: specify hardness (bigger is harder)
  - `--tau_plus`: specify class probability
  - `--batch_size`: batch size for SimCLR

For instance, run the following command to train an embedding on STL10 with hard negatives.
```
python main.py --tau_plus 0.1 --beta 1.0
```

## Linear evaluation
The model is evaluated by training a linear classifier after fixing the learned embedding.

path flags:
  - `--model_path`: specify the path to saved model
```
python linear.py --model_path results/model.pth
```

An example of both training an embedding and evaluating can be run using:
```
bash launch.sh
```

#### Pretrained Models
|          | beta | tau_plus | Arch | Latent Dim | Batch Size  | Accuracy(%) | Download |
|----------|:---:|:---:|:----:|:---:|:---:|:---:|:---:|
|  Biased | beta = 0.0 | tau_plus = 0.0 | ResNet50 | 128  | 256  | 80.15  |  [model](https://drive.google.com/file/d/1qQE03ztnQCK4dtG-GPwCvF66nq_Mk_mo/view?usp=sharing)|
|  Debiased | beta = 0.0 | tau_plus = 0.1 | ResNet50 | 128  | 256  | 84.26  |   [model](https://drive.google.com/file/d/1d8nfGHsHIuJYjU7mHtCtSXf98IbWMFAa/view?usp=sharing)|
|  Hard | beta = 0.5 | tau_plus = 0.1 | ResNet50 | 128  | 256  | 86.38 |  [model](https://drive.google.com/file/d/1-XgYrqiBPswCDIMAWAFskUjsiaXEpibu/view?usp=sharing)|
|  Hard | beta = 1.0 | tau_plus = 0.1 | ResNet50 | 128  | 256  | 87.44 |  [model](https://drive.google.com/file/d/1l-PMUzdK0pXZDjMLdFmIIy07DWiQ5Mm4/view?usp=sharing)|
|  Hard | beta = 2.0 | tau_plus = 0.1 | ResNet50 | 128  | 256  | 87.09 |  [model](https://drive.google.com/file/d/1EEnUp4DJpHR7NRgQdAZgZoYWw5ZajsFv/view?usp=sharing)|
|  Hard | beta = 6.0 | tau_plus = 0.1 | ResNet50 | 128  | 256  | 83.09 |  [model](https://drive.google.com/file/d/1LynFacKy_iaFrF6Qi2hRXHFGSEiY1Wis/view?usp=sharing)|

## Acknowledgements

Part of this code is inspired by [leftthomas/SimCLR](https://github.com/leftthomas/SimCLR) and [chingyaoc/DCL](https://github.com/chingyaoc/DCL).
