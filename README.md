# Contrastive Learning with Hard Negative Samples

<p align='center'>
<img src='https://github.com/joshr17/HCL/blob/main/figs/hard_sampling_schema.png?raw=true' width='800'/>
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


**Contrastive Learning with Hard Negative Samples** [[paper]](https://arxiv.org/pdf/2010.04592)
<br/>
[Joshua Robinson](https://joshrobinson.mit.edu/), 
[Ching-Yao Chuang](https://chingyaoc.github.io/), 
[Suvrit Sra](http://web.mit.edu/torralba/www/), and
[Stefanie Jegelka](https://people.csail.mit.edu/stefje/)
<br/>


## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{robinson2020hard,
  title={Contrastive Learning with Hard Negative Samples},
  author={Robinson, Joshua and Chuang, Ching-Yao, and Sra, Suvrit and Jegelka, Stefanie},
  journal={arXiv:2010.04592},
  year={2020}
}
```
For any questions, please contact Josh Robinson (joshrob@mit.edu).

## Acknowledgements

Part of this code is inspired by [leftthomas/SimCLR](https://github.com/leftthomas/SimCLR), by [chingyaoc/DCL](https://github.com/chingyaoc/DCL), and by [fanyun-sun/InfoGraph](https://github.com/fanyun-sun/InfoGraph).
