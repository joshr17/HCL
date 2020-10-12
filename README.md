# Contrastive Learning with Hard Negatives Samples
Method for sampling hard negatives for contrastive learning


# Debiased Contrastive Learning

<p align='center'>
<img src='https://github.com/chingyaoc/DCL/blob/master/misc/fig1.png?raw=true' width='500'/>
</p>

We consider the question: how can you sample good negative examples for contrastive
learning? We argue that, as with metric learning, learning contrastive representations
benefits from hard negative samples (i.e., points that are difficult to distinguish from
an anchor point). The key challenge toward using hard negatives is that contrastive
methods must remain unsupervised, making it infeasible to adopt existing negative
sampling strategies that use label information. In response, we develop a new class
of unsupervised methods for selecting hard negative samples where the user can
control the amount of hardness. A limiting case of this sampling results in a representation that tightly clusters each class, and pushes different classes as far apart as possible. The proposed method improves downstream performance across multiple
modalities, requires only few additional lines of code to implement, and introduces no
computational overhead.


**Debiased Contrastive Learning** [[paper]](https://arxiv.org/abs/2007.00224)
<br/>
[Ching-Yao Chuang](https://chingyaoc.github.io/), 
[Joshua Robinson](https://joshrobinson.mit.edu/), 
[Lin Yen-Chen](https://yenchenlin.me/),
[Antonio Torralba](http://web.mit.edu/torralba/www/), and
[Stefanie Jegelka](https://people.csail.mit.edu/stefje/)
<br/>


## Prerequisites
- Python 3.7 
- PyTorch 1.3.1
- PIL
- OpenCV

## Contrastive Representation Learning
We can train standard (biased) or debiased version (M=1) of [SimCLR](https://arxiv.org/abs/2002.05709) with `main.py` on STL10 dataset.

flags:
  - `--debiased`: use debiased objective (True) or standard objective (False)
  - `--tau_plus`: specify class probability
  - `--batch_size`: batch size for SimCLR

For instance, run the following command to train a debiased encoder.
```
python main.py --tau_plus = 0.1
```

## Linear evaluation
The model is evaluated by training a linear classifier after fixing the learned embedding.

path flags:
  - `--model_path`: specify the path to saved model
```
python linear.py --model_path results/model_400.pth
```

#### Pretrained Models
|          | tau_plus | Arch | Latent Dim | Batch Size  | Accuracy(%) | Download |
|----------|:---:|:----:|:---:|:---:|:---:|:---:|
|  Biased | tau_plus = 0.0 | ResNet50 | 128  | 256  | 80.15  |  [model](https://drive.google.com/file/d/1qQE03ztnQCK4dtG-GPwCvF66nq_Mk_mo/view?usp=sharing)|
|  Debiased |tau_plus = 0.05 | ResNet50 | 128  | 256  | 81.85  |  [model](https://drive.google.com/file/d/1pA4Hpcug8tbgH9O6PCu-447vJzxbbR5I/view?usp=sharing)|
|  Debiased |tau_plus = 0.1 | ResNet50 | 128  | 256  | 84.26  |   [model](https://drive.google.com/file/d/1d8nfGHsHIuJYjU7mHtCtSXf98IbWMFAa/view?usp=sharing)|

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{chuang2020debiased,
  title={Debiased Contrastive Learning},
  author={Chuang, Ching-Yao and Robinson, Joshua and Yen-Chen, Lin and Torralba, Antonio and Jegelka, Stefanie},
  journal={arXiv preprint arXiv:2007.00224},
  year={2020}
}
```
For any questions, please contact Ching-Yao Chuang (cychuang@mit.edu).

## Acknowledgements

Part of this code is inspired by [leftthomas/SimCLR](https://github.com/leftthomas/SimCLR).
