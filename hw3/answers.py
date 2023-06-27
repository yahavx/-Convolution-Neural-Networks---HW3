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
    start_seq = "To be, or not to be: that is the question."
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We split the corpus into sequences instead of training on the entire text for several reasons.
First for memory reasons because loading an entire text can consume memory so significantly
that it will not be possible to load the entire text and train. Keep in mind that sometimes
our processing capabilities are limited. Another reason, is more effective training, so train
a batch of samples when each sample is a sequence. Another reason, following on from the
previous one, that is more relevant to word-level RNN is the fact that the GRU's memory
capacity is limited (that's why we got the LSTM and later the Transformers) and training
on a long sequence can damage the training ability because the gradients are getting weaker
and the ability to correct distant predictions are getting smaller.
"""

part1_q2 = r"""
In fact, the hidden state can be used repeatedly (by re-feeding it to the network) to generate
additional character predictions and do so cyclically for whatever length we want.
The hidden state encodes the system's memory, so it can be used to generate additional characters.
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
1. We lower the temperature for sampling because in this way (as we saw in the graph) we get a
distribution on the characters that is less and less close to a uniform distribution.
This way we can increase the chance of sampling the char(s) with the highest scores (probabilities)
compared to the others and get better sequences (hopefully). </br>
2. When the temperature is very high, the sampling distribution becomes more and more close to
uniform and thus the characters are chosen randomly. It happens because we get $\vec{y}/T=0$ when
$T$ increases and then $e^{\vec{y}/T}=1$ for every char. If the scores for softmax are equal, 
we get a uniform distribution $\mathrm{softmax}_T(\vec{y}) = \frac{1}{k}$.</br>
3. When the temperature is very low, the sampling distribution becomes less and less close to
uniform and thus the characters are chosen based on the model's predictions according to the
highest probabilities. In fact the function emphasizes the differences between the chars.
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
