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
        'training_norm_params' :  _root + r"\cvae_diffusion\dataset\training\stats.json",
        'validation_norm_params' :  _root + r"\cvae_diffusion\dataset\validation\stats.json",
        'checkpoint': _root + r"\cvae_diffusion\checkpoint.pth",
    },
    
    'autoencoder_diffusion' : {
        'model': _root + r"\autoencoder_diffusion\model.pth",
        'config': _root + r"\autoencoder_diffusion\config.json",
        'scheduler': _root + r"\autoencoder_diffusion\scheduler.pth",
        'dataset_training' : _root + r"\autoencoder_diffusion\dataset\training.pt",
        'dataset_validation' : _root + r"\autoencoder_diffusion\dataset\validation.pt",
        'training_norm_params' :  _root + r"\autoencoder_diffusion\dataset\training_params.pt",
        'validation_norm_params' :  _root + r"\autoencoder_diffusion\dataset\validation_params.pt",
        'checkpoint': _root + r"\autoencoder_diffusion\checkpoint.pth",
    },
    'augmented_data_1' : _root + r"\augmented_1"
    
}

VAE_LATENT_DIM = 64
VAE_CHANNELS = [2, 16, 32, 64, 128]
VAE_STRIDES = [(2,2), (2,2), (2,1), (1,1)]

cVAE_LATENT_DIM = 128
cVAE_CHANNELS = [1, 32, 64, 128, 256]