# Reverse-Engineering a Transformer on Modular Arithmetic

This project investigates how a small transformer model performs modular addition by reverse-engineering its internal computation. The goal is to move beyond input–output behaviour and uncover the *mechanistic* structure of the model’s reasoning process.

## Project Overview

* Trained (or used a pretrained) transformer on modular arithmetic tasks.
* Reverse-engineered the model to understand **how** it represents and composes addition operations.
* Analysed components including:

  * **Attention heads** – information routing and token interaction patterns
  * **Feature circuits** – latent representations responsible for carrying arithmetic structure
  * **Activation pathways** – step-by-step flows that show how intermediate computations emerge
