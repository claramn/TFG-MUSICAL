
import torch.nn as nn       #contiene los modulos basicos para crear redes neuronales
from torch.distributions import Normal      #Normal se usa para muestrear el espacio latente en el VAE
from src.utils.models import *      

"""
import torch.nn as nn
from torch.distributions import Normal
from src.utils.models import *
import torch.nn.functional as F
import torch
"""
"""
detalles a entender:
CAPAS CONVOLUCION
toman una entrada (pej un espectrograma) y aplica varios filtros (kernels) q se desplazan sobre ella.
cada filtro detecta un tipo d patron local: bordes, texturas, transiciones d frecuencia etc.
cada filtro busca un tipo d forma en la imagen, cuantos mas filtros, mas tipos d patrones aprende el modelo 

CAPAS CONVOLUCION TRANSPUESTA
son lo opuesto a las normales. en lugar d reducir el tam lo aumentan, sirven para reconstruir o generar datos a partir d 
una representacion comprimida. expanden el mapa latente a su og form

backpropagation: es el algoritmo q entrena la red neuronal. Permite ajustar los pesos de las neuronas para q la salida d la red se acerque lo más posible a la deseada.
Es un método para calcular como cambia el error total d la red cuando cambias cada peso, y usar eso para corregirlos.
Con el loss q nos sale, la red mira cuanto se ha equivocado y calcula como cada peso contribuyo a ese error.´

REPARAMETRIZACION (VAE)
en un vae el encoder no genera un unico punto en el espacio latente, sino una distribución gaussiana. D ahí se debe 
muestrear un vector z (el q pasará al decoder)

pero muestrear directamente rompería el flujo d gradientes, no podríamos entrenar la red con backpropagation, pq el encoder
no devuelve un solo vector fijo, sino parametros d una distribucion. La red aprende d una media y una varianza para cada entrada.
Luego se muestra un valor z d esa distribucion q pasa al decoder para reconstruir la entrada. Hay que separar el ruido del aprendizaje. 
genera un vector entrenable 

DIVERGENCIA KL
mide cuanto se diferencia una distribucion d otra. En el VAE mide la distancia entre la distribucion latente aprendida
y la distribucion prior. queremos que ambas se parezcan para q el modelo pueda generar nuevos datos dsd un ruido gaussiano estandar

ELBO
(evidence lower bound)
la funcion objetivo q entrena un VAE. se deriva del teorema d bayes, y combina qué tan bien el modelo reconstruye los datos con
la penalización d desviarse d la distribucion normal estandar
Reconstrucción → que la salida se parezca al original.

KL → que el espacio latente sea “ordenado” y continuo.
El equilibrio entre ambos hace que el VAE genere datos realistas y variados.

"""

# esta clase toma una entrada, pej un espectrograma y la codifica en un vector latente
class Encoder(nn.Module):
    """
    input_size: dimensiones iniciales (alto,ancho) del espectrograma
    latent_dim: tamaño del vector latente
    channels: lista con el num d canales d cada capa convolucional (ej [1,32,64,128])
    variational: si es true, genera media y varianza (para VAE)
    """
    def __init__(self, input_size, latent_dim, channels, variational=False):
        super().__init__()

        # Is the encoder serving an autoencoder or a variational autoencoder?
        self.variational = variational

        #parametros d las convoluciones, define las caracteristicas de cada convolucion 2D: filtros 3x3, stride 2(reduce resolucion a la mitad), padding 1 (para mantener tamaños consistentes)
        conv_kernel_size = (3, 3)
        conv_stride = (2, 2)  # Stride 2 for downsampling
        conv_padding = (1, 1)

        #construccion d las capas convolucionales
        """
        va creando secuencialmente capas conv2d
        calcula el nuevo tamaño despues d cada convolucion con la funcion auxiliar compute_conv2d_output_size()
        """
        self.sizes = [input_size]
        current_size = input_size

        blocks = []
        for i in range(1, len(channels)):
            """
            in_channels: numero d canales d enrada (pej 1 si es escala d grises o espectrograma)
            out_channels: num d filtros q se aplican (mas filtros mas profundidad d caracteristicas)
            kernel_size: tamaño de cada filtro
            stride=2: cada filtro salta 2 pixeles -> reduce resolucion
            padding=1: añade borde d ceros para no perder tamaño al aplicar los filtros

            el encoder va reduciendo tamaño pero aumentando el numero d canales, osea menos info espacial pero mas info semantica 
            """
            conv = nn.Conv2d(channels[i - 1], channels[i], kernel_size=conv_kernel_size, stride=conv_stride,padding=conv_padding)
            current_size = compute_conv2D_output_size(current_size, conv_kernel_size, conv_stride, conv_padding)
            self.sizes.append(current_size)

            #añade activaciones y normalizacion. En un VAE no se activa BatchNorm pq puede interferir con la distribucion probabilistica
            if variational: # BatchNorm can hurt VAE
                block = nn.Sequential(conv, nn.ReLU())
            else:
                block = nn.Sequential(conv, nn.ReLU(), nn.BatchNorm2d(channels[i]))

            blocks.append(block)

        #ensamblado final del encoder
        """
        fc1: capa lineal final para el vector latente (o la media, en un VAE)
        fc2: capa lineal adicional para la varianza logaritmica en el caso variacional
        """
        self.encoder = nn.Sequential(*blocks)

        self.flatten = nn.Flatten()

        # If varational=True, fc1 represents the mean layer
        self.fc1 = nn.Linear(channels[-1] * current_size[0] * current_size[1], latent_dim)

        # fc2 represents the log_var layer
        self.fc2 = nn.Linear(channels[-1] * current_size[0] * current_size[1], latent_dim)

    """
    en un AE: devuelve directamente el vector comprimido
    en un VAE: devuelve la media y la varianza logarítmica para muestrear el espacio latente
    """
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)

        if self.variational:
            mu = self.fc1(x)
            log_var = self.fc2(x)
            return x, mu, log_var
        
        x = self.fc1(x)
        return x

    def get_sizes(self):
        return self.sizes

"""
reconstruye el input original a partir del vector latente
recibe los tamaños calculados en el encoder para poder revertir la arquitectura
"""
class Decoder(nn.Module):
    def __init__(self, sizes, latent_dim, channels, variational=False):
        super().__init__()

        kernel_size = (3, 3)
        stride = (2, 2)
        padding = (1, 1)
        output_padding = 0

        """
        Preparacion del vector latente
        expande el vector latente para que tenga forma d mapa d caracteristicas
        prepara los tamaños invertidos del encoder para reconstruir paso a paso
        """
        rev_channels = list(reversed(channels))
        rev_sizes = list(reversed(sizes))

        expected_size = rev_sizes[0]
        self.fc = nn.Linear(latent_dim, rev_channels[0] * expected_size[0] * expected_size[1])
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(rev_channels[0], expected_size[0], expected_size[1]))

        deconv_blocks = []

        """
        Construccion d capas convtranspose2d
        usa convulciones transpuestas para deshacer las convoluciones del encoder
        ReLU se aplica salvo en la ultima capa del VAE (para permitir valores negativos)

        stride=2: duplica el tam espacial
        se usan para deshacer la compresion del encoder
        ReLU: introduce no linealidad en la red, necesaria para q aprenda funciones complejas
        ReLU(x)=max(0,x)
        si el valor es positivo lo deja igual, sino lo convierte en 0
        Hace q las redes sean mas estables al entrenar, evita ssaturaciones 
        """

      #  current_size = expected_size    #empieza en rev_sizes[0]

        for i in range(1, len(rev_sizes)):
            target = rev_sizes[i]  # el tamaño que queremos alcanzar en esta capa
            calc_no_op = compute_convTranspose2D_output_size(
                current_size, kernel_size, stride, padding
            )   #cálculo del size esperado sin output_padding

            #decidir output padding por eje (0 o 1)
            op_h = target[0] - calc_no_op[0]
            op_w = target[1] - calc_no_op[1]
            op_h = 1 if op_h > 0 else 0
            op_w = 1 if op_w > 0 else 0
            output_padding = (op_h, op_w)
            #opcional: debug para ver tamaños capa a capa
            
            #crear la capa transposta con ese output padding, antes output padding era siempre 0
            deconv = nn.ConvTranspose2d(
                rev_channels[i - 1],
                rev_channels[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            )

            # We don't add a ReLU in last layer in the case of a Variational Decoder
            if i < len(rev_sizes) - 1 or not variational:
                block = nn.Sequential(deconv, nn.ReLU())
            else:
                block = nn.Sequential(deconv)

            deconv_blocks.append(block)
            
            #actualizar current_size para la siguiente capa. nuestro helper no acepta output padding, recomputa a mano
            current_size = (
                calc_no_op[0] + output_padding[0],
                calc_no_op[1] + output_padding[1],
            )
            
            assert current_size == target, f"Mismatch en capa {i}: got {current_size} vs target {target}"
        self.decoder = nn.Sequential(*deconv_blocks) #, nn.Signmoid())

    #devuelve la reconstruccion (una imagen/espectograma del mismo tamaño q la entrada original)
    def forward(self, x):
       # print("[DEC FWD] input z:", x.shape)
        x = self.fc(x)
        x = self.unflatten(x)
        ##print("[DEC FWD] after unflatten:", x.shape)
        x = self.decoder(x)
       # print("[DEC FWD] after convT stack:", x.shape)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_dim, channels=None):
        super().__init__()

        if not channels:
            raise ValueError('channels argument in AutoEncoder class must be valid')

        self.input_size = input_size
        self.channels = channels

        self.encoder = Encoder(input_size, latent_dim, channels)
        sizes = self.encoder.get_sizes()
      #  print("[ENCODER] sizes:", sizes)    #para ver los tamaños, t dice como van quedando las h, w en cada capa
        self.decoder = Decoder(sizes, latent_dim, channels)

    #reconstruye y ajusta el tamaño final por si hay diferencias d pixeles debido a padding
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        #print("[AE FWD] x:", x.shape, "rec:", reconstructed.shape)

        target_height, target_width = x.shape[2], x.shape[3]
        reconstructed = adjust_shape(reconstructed, (target_height, target_width))
        return reconstructed

class VAE(nn.Module):
    def __init__(self, input_size, latent_dim, channels=None):
        super().__init__()
        
        if not channels:
            raise ValueError('channels argument in VAE class must be valid')

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.channels = channels

        #define una distribucion gaussiana estandar en GPU para muestrear
        self.normal = Normal(0, 1)
        self.normal.loc = self.normal.loc.cuda()
        self.normal.scale = self.normal.scale.cuda()

        self.encoder = Encoder(input_size, latent_dim, self.channels, variational=True)
        sizes = self.encoder.get_sizes()
        self.decoder = Decoder(sizes, latent_dim, self.channels, variational=True)  

    #implementa el truco d reparametrizacion para permitir el backpropagation a traves del muestreo
    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = self.normal.sample(mean.shape)
        Z = mean + eps * std
        return Z.to(device=std.device)
    
    #si el encoder se aleja demasiado d una normal estandar, el KL crece -> penalizacion
    def compute_kld(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=-1)
        return kld

    def calculate_ELBO_terms(self, x, sigma=1.0):
        B, C, H, W = x.shape

        # Encode to get posterior parameters
        _, mean, log_var = self.encoder(x)
        log_var = torch.clamp(log_var, min=-20, max=20)

        # Sample z ~ q(z|x)
        z = self.reparameterization(mean, log_var)

        # Decode to reconstruction
        x_hat = self.decoder(z)
        x_hat = adjust_shape(x_hat, (H, W), pad_mode='reflect')  # [B,C,H,W]

        # Reconstruction term
        recon = - 1/(2*sigma**2) * F.mse_loss(x_hat, x, reduction='none').view(B, -1).sum(dim=1)

        # KL divergence term
        kld = self.compute_kld(mean, log_var) # (B,)

        return recon, kld

    def forward(self, x):
        return self.calculate_ELBO_terms(x)

    def loss_function(self, recon, kld):
        # We return -ELBO, since we'retrying to maximize ELBO
        return (-recon + kld).mean()

