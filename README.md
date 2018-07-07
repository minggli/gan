# mnist-gan
generate images of handwriting digits using MNIST toy dataset.

Generative Adversarial Networks (GAN) is by large a framework of adversarial learning between Discriminator D(x) and Generator G(z) that aims to achieve Nash equilibrium between D and G where G(z) should successfully approximate P(x) and therefore generate realistic samples.

GAN was introduced by [Goodfellow et al 2014][1]. It was observed in subsequent research (e.g. [Arjovsky and Bottou 2017][2]) the difficult and unstable training of GANs. Two main reasons are: First, distributions P(x) and G(z) can be disjoint with supports in respective low dimensional manifolds that rarely intersect. Thus a perfect discriminator exists separating disjoint G(z) and P(x), causing gradient-based method to fail to learn and recover; Secondly, the original choice of Kullback–Leibler divergence as cost function yields large if not infinite loss even when P(x) and G(z) are close.

Gaussian noise term applied to both D(x) and G(z) and Wasserstein distance are argued in [Arjovsky and Bottou 2017][2] to soften measurement of similarity and Wasserstein GAN is formally discussed in [Arjovsky et al 2017][3].

[Gulrajani et al 2017][4] further improves the continuity with use of Gradient Penalty (GP)

This repository is a toy project to explore the maturity of WGAN-GP and hopefully motivate for wider application.

[1]: https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf  
[2]: https://arxiv.org/abs/1701.04862  
[3]: https://arxiv.org/abs/1701.07875  
[4]: https://arxiv.org/abs/1704.00028
