_root = r"C:\Users\Articuno\Desktop\TFG-MUSICAL\data"

# TODO usar los paths estos
PATHS = {
    'autoencoder' : {
        'model' : _root + r"\models\autoencoder_fase.pth"
    },
    'autoencoder_2D' : {
        'model' : _root + r"\models\latent_autoencoder.pth"
    },
    'VAE' : {
        'model' : _root + r"\models\vae.pth"
    },
    'VAE_2D' : {
        'model' : _root + r"\models\latent_VAE.pth",
        'checkpoint' : _root + r"\models\latent_VAE_check.pth"
    },
    'cVAE_2D' : {
        'model' : _root + r"\models\cvae7_latent.pth",
        'checkpoint' : _root + r"\models\latent_cVAE_check.pth"
    },
    'minst_VAE' : {
        'model' : _root + r"\models\minst.pth"
    },
    'minst_diffusion' : {
        'model': _root + r"\minst_diffusion\model.pth",
        'config': _root + r"\minst_diffusion\config.json",
        'scheduler': _root + r"\minst_diffusion\scheduler.pth",
        'checkpoint' : _root + r"\minst_diffusion\checkpoint.pth",
        'dataset' : r"C:\Users\Articuno\Desktop\TFG-info\data\mnist"
    },
    'diffusion' : {
        'model': _root + r"\diffusion\model.pth",
        'config': _root + r"\diffusion\config.json",
        'scheduler': _root + r"\diffusion\scheduler.pth",
        'checkpoint': _root + r"\diffusion\checkpoint.pth"
    },
    'vae_diffusion' : {
        'model': _root + r"\vae_diffusion\model.pth",
        'config': _root + r"\vae_diffusion\config.json",
        'scheduler': _root + r"\vae_diffusion\scheduler.pth",
        'dataset_training' : _root + r"\vae_diffusion\dataset\training",
        'dataset_validation' : _root + r"\vae_diffusion\dataset\validation",
        'dataset_testing' : _root + r"\vae_diffusion\dataset\testing",
        'training_norm_params' :  _root + r"\vae_diffusion\dataset\training\stats.json",
        'validation_norm_params' :  _root + r"\vae_diffusion\dataset\validation\stats.json",
        'checkpoint': _root + r"\vae_diffusion\checkpoint.pth",
    },
    'cvae_diffusion' : {
        'model': _root + r"\cvae_diffusion\model.pth",
        'config': _root + r"\cvae_diffusion\config.json",
        'scheduler': _root + r"\cvae_diffusion\scheduler.pth",
        'dataset_training' : _root + r"\cvae_diffusion\dataset\training",
        'dataset_validation' : _root + r"\cvae_diffusion\dataset\validation",
        'dataset_testing' : _root + r"\cvae_diffusion\dataset\testing",
        'training_norm_params' :  _root + r"\cvae_diffusion\dataset\training\stats.json",
        'validation_norm_params' :  _root + r"\cvae_diffusion\dataset\validation\stats.json",
        'checkpoint': _root + r"\cvae_diffusion\checkpoint.pth",
    },
    'autoencoder_diffusion' : {
        'model': _root + r"\autoencoder_diffusion\model.pth",
        'config': _root + r"\autoencoder_diffusion\config.json",
        'scheduler': _root + r"\autoencoder_diffusion\scheduler.pth",
        'dataset_training' : _root + r"\autoencoder_diffusion\dataset\training",
        'dataset_validation' : _root + r"\autoencoder_diffusion\dataset\validation",
        'dataset_testing' : _root + r"\autoencoder_diffusion\dataset\testing",
        'training_norm_params' :  _root + r"\autoencoder_diffusion\dataset\training_params.pt",
        'validation_norm_params' :  _root + r"\autoencoder_diffusion\dataset\validation_params.pt",
        'checkpoint': _root + r"\autoencoder_diffusion\checkpoint.pth",
    },
    'augmented_data_1' : _root + r"\augmented_1"
    
}

# VAE parameters
VAE_LATENT_DIM = 64
VAE_CHANNELS = [2, 16, 32, 64, 128]
VAE_STRIDES = [(2,2), (2,2), (2,1), (1,1)]
VAE_INPUT_HEIGHT = 1500
VAE_INPUT_WIDTH = 251
VAE_INPUT_SIZE = (VAE_INPUT_HEIGHT, VAE_INPUT_WIDTH)

# cVAE parameters
CVAE_LATENT_DIM = 128
CVAE_CHANNELS = [1, 32, 64, 128, 256]
CVAE_CONDITION_DIM = 128
CVAE_N_HARMONICS = 64
CVAE_DDSP_HIDDEN = 256
CVAE_N_MELS = 80
CVAE_MAX_FRAMES = 128
CVAE_INPUT_SIZE = (CVAE_N_MELS, CVAE_MAX_FRAMES)

AE_INPUT_HEIGHT = 4000 # = n_fft because onesided=False
AE_INPUT_WIDTH = 201 # = 1 + floor((T - n_ftt) / hop_length), where T is the whole duration (4 * 16000 = 64000)
AE_LATENT_DIM = 200 # TODO seguramente deba cambiarlo a 8
AE_CHANNELS = [2, 16, 32, 64]
AE_INPUT_SIZE = (AE_INPUT_HEIGHT, AE_INPUT_WIDTH)