# VAE Examples

This directory contains the outputs of several VAE training runs on the NSynth spectrogram dataset. Each subfolder corresponds to one experiment and includes:

- **configs.txt**  
  JSON-formatted hyperparameters and training settings.

- **examples/**  
  Five audio clips sampled from the latent space (`.wav` files).

- **losses.png**  
  Training vs. validation loss curves over epochs.

- **PCA.png**  
  Two-dimensional PCA projection of the encoder’s latent means.

- **term_losses.png**  
  Reconstruction (MSE) vs. KL divergence plotted over training.

- **beta_vs_kl.png** *(optional)*  
  Annealing weight β vs. KL divergence over epochs (present only for experiments that logged β schedules).
