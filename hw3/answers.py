r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============


# ==============
# Part 2 answers


def part2_vae_hyperparams():
    # hypers = dict(
    #     batch_size=0,
    #     h_dim=0, z_dim=0, x_sigma2=0,
    #     learn_rate=0.0, betas=(0.0, 0.0),
    # )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=10,
        h_dim=256, z_dim=128, x_sigma2=0.5,
        learn_rate=0.001, betas=(0.9, 0.999),
    )
    # ========================
    return hypers


part2_q1 = r"""
$\sigma^2$ determines the balance between the impact of the reconstruction loss and the KL divergence term. 
When it's higher, the reconstruction influences less on the overall loss, so during optimization, the emphasis shifts
towards minimizing the KL divergence from the prior distribution.
In the other case, when it's smaller, the reconstruction influences more on the overall loss, so during optimization,
we will focus more on minimizing the reconstruction loss.
"""

part2_q2 = r"""
1. The purpose of the reconstruction loss is to encourage the model to generate outputs that are simliar to the inputs, which is our main task.
The purpose of the KL divergence loss is to avoid overfitting (regularization), by encouraging the latent space to approximate the chosen prior distribution (usually normal distribution).

2. The loss term of the KL divergence is determined by the distance between the latent-space distribution to the prior distribution. In other words, this term penalizes the latent-space distribution 
of the model for not being close to the prior distribution.

3. This is a regularization term, the benefit of it is that it helps avoiding overfitting. Without it, the model would tend to find representations that works well with the training data, but it wouldn't be able to generalize to unseen data.
"""

# ==============


# ==============
# Part 3 answers


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
