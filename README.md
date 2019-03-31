# Deterministic-GAIL-PyTorch
This is an attempt to implement Generative Adversarial Imitation Learning (GAIL) for deterministic policies with off Policy learning. The algorithm never interacts with the environment (except for evaluation), instead it is trained on policy state-action pair, where policy only selects actions for states sampled from expert data.


## Results

Although it works sometimes (depending on the type of environment), the algorithm has very high variance, and the results are highly inconsistent.

### BipedalWalker-v2

