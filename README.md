# Deterministic-GAIL-PyTorch
This is an attempt to implement Generative Adversarial Imitation Learning (GAIL) for deterministic policies with off Policy learning. The algorithm never interacts with the environment (except for evaluation), instead it is trained on policy state-action pair, where **policy only selects actions for states sampled from expert data**.


## Results

Although it works sometimes (depending on the type of environment), the algorithm has high variance, and the results are inconsistent.

### BipedalWalker-v2

Expert Policy              |  Recovered Policy (10 expert episodes)
:-------------------------:|:-------------------------:
![](https://github.com/nikhilbarhate99/Deterministic-GAIL-PyTorch/blob/master/gif/BipedalWalker_expert.gif) |  ![](https://github.com/nikhilbarhate99/Deterministic-GAIL-PyTorch/blob/master/gif/BipedalWalker_learned.gif)

Epochs vs rewards   |
:-------------------------:
![](https://github.com/nikhilbarhate99/Deterministic-GAIL-PyTorch/blob/master/gif/graph_BipedalWalker-v2.png)
