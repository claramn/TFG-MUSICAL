"""
esto es importante para el ddsp pq el ddsp hace esto : controles interpretables -> sintetizador -> audio
y aqui se simplifica haciendo: f0 + loudness -> senoide -> audio
mas adelante le podemos meter armonicos y ruidio para q el audio sea mas realista 
BASICAMENTE estamos desmontando un audio en parametros simples y reconstruyendolos dsd los parametros para comporbar q el pipeline d analisis + sintesis funciona
"""

import sys
sys.path.append("src")

import math
import torch
import torchaudio
import torchcrepe
from dataset import NSynth

#el codigo corre en cuda (GPU envidia) y cpu. Guardamos cuda o cpu en device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# cargar muestra usando la particion training
"""
waveform: audio como tensor
sr: sample rate
key: identificador textual del sample
metadata: info extra, como el instrumento
"""
ds = NSynth("training")
waveform, sr, key, metadata = ds[0] #pide el primer sample del dataset

"""
aseguramos audio mono. A veces un audio puede tener más d un canal. waveform.shape mira cuantos canales tiene. solo podemos trabajar con audio mono, por lo q si no lo es lo cambiamos para q si 
"""
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

audio = waveform.to(device)
n_samples = audio.shape[-1] #saca el num total d muestras d audio. si el audio es d 4 s a 16kHz, 4*16000=64000 muestras

# extraer f0
hop_length = int(sr / 200.0)    #cuanto tiempo queremos una estimacion d pitch, si sr = 16000, 16000/200=80 -> cada 80 samples, q son 80/16000 = 0.05 s, osea 5 ms or frame, lo q quiere decir q queremos estimar cada 5 milisegundos
"""
el predict recibe un audio y t devuelve una estimacion del pitch
La periodicity nos dice si un frame parece tener una estructura periódica clara. si es alta, el pitch es probablemente fiable, si no lo es, ese frame puede ser ruido o mala estimacion
"""
pitch = torchcrepe.predict(
    audio, #waveform d entrada
    sr, 
    hop_length, #cada cuantos samples queremos una prediccion
    32.70,  #frecuencia minima buscada, esta mas o menos es un C1
    1975.53,    # frec max buscada
    model="tiny",
    batch_size=512, #cuantos fragmentos se procesan a la vez
    device=device,
    return_preiodicity=True #ademas del pitch, nos da una medida d fiabilidad
)

# limpiar pitch
"""
torchcrepe devuelve [batch,frames]
como solo usamos un audio, batch = 1. squeeze elimina esa dimension extra, entonces pasamos d [1,801] a [801], q es mas comodo para trabajar
"""
pitch = pitch.squeeze()
periodicity = periodicity.squeeze(0) # [frames]

"""
a veces pueden salir valores raros, asi q los convertimso en 0
"""
pitch = torch.nan_to_num(pitch, nan=0.0, posinf=0.0, neginf=0.0)
periodicity = torch.nan_to_num(periodicity, nan=0.0, posinf=0.0, neginf=0.0)

# Filtrar frames poco fiables
pitch = torch.where(periodicity > 0.7, pitch, torch.zeros_like(pitch))
"""
pitch[pitch > 0] se queda solo con frames validos
valid.numel cuenta cuantos elementos hay
valid.mean.item calcula la media del pitch valido
valid.median.item calcula la mediana
esto nos dice si el pitch tiene sentido. la mediana suele ser mas estable q la media cuando hay pitch locos
"""
valid = pitch[pitch > 0]
if valid.numel() > 0:
    print("pitch medio filtrado (Hz):", valid.mean().item())
    print("pitch mediano filtrado (Hz):", valid.median().item())
else:
    print("no se detectó pitch válido")
    
# interpolar pitch a resolución de sample
"""
rn pitch esta definido por frames, pej 801 valores
para sintetizar audio sample a sample necesitamos algo del tamaño del audio completo: 64000 valores
estiramos la curva de pitch para tener un valor por sample
entre dos frames d pitch, rellena valores intermedios suavemente
"""
pitch_up = torch.nn.functional.interpolate(
    pitch[None, None, :],   #añade dos dimensiones para q interpolate acepte el formato, pasa d [frames] a [1,1,fra mes]
    size=n_samples, #queremos q la salida tenga longitud igual al numero d samples del audio 
    mode="linear",  #interpolacion lineal
    align_corners=False
).squeeze() #quita dimensiones sobrantes y deja un vector plano

# evitar zonas en 0 para que no explote la fase. esto es para evitar q la integracion d fase pete
pitch_up = torch.clamp(pitch_up, min=1.0)


# frame_length ~ 1024 para 16kHz. definimos el tam d ventana para calcular energia. cada ventana tendra 1024 samples
frame_length = 1024

# energia RMS por frame
rms = audio.unfold(-1, frame_length, hop_length)          # [1, n_frames, frame_length]. unfold corta el audio en ventanas, si el audio es largo, esto lo convierte en muchas ventanitas. -1 usa la ultima dimension, o sea el eje temporal. entre una ventana y la siguiente se avanza hop length
"""
calculamos el rms en cada ventana. rms= root mean square
es una medida d energia, amplitud media. 
rms**2: eleva al cuadrado los samples
.mean hace la media dentro d cada ventana
+ 1e-8 evita problemas numericos
Nos da una medida d cuanta fuerza tiene el audio en cada frame
"""
rms = torch.sqrt(torch.mean(rms ** 2, dim=-1) + 1e-8)     # [1, n_frames]
rms = rms.squeeze(0)    #quita dimension batch


# normalizar a [0,1]. escala la energia para q sea 1 aprox, para q la envolvente queda en un rango comodo [0,1]
rms = rms / (rms.max() + 1e-8)

# interpolar loudness a sample-rate
"""
pasamos d una energia por frame a una por sample, limitamos el rango para q la envolvente este entre 0 y 1
"""
rms_up = torch.nn.functional.interpolate(
    rms[None, None, :],
    size=n_samples,
    mode="linear",
    align_corners=False
).squeeze()

# suavizado opcional
rms_up = torch.clamp(rms_up, min=0.0, max=1.0)


# sintetizar senoide con envolvente
"""
construye la fase acumulada d la senoide, sin(fase), la fase depende d la frecuencia.
pitch_up/sr: convierte frecuencia en incremento por sample
torch.cumsum: hace suma acumulada, va integrando la frecuencia a lo largo del tiempo
2*math.pi*...: convierte eso a radianes, q es lo q usa sin
esto crea una senoide cuya frecuencia instantanea sigue pitch_up
"""
phase = 2 * math.pi * torch.cumsum(pitch_up / sr, dim=0)
"""
sintetizamos la señal final, creando una snoide. rms_up hace q la amplitud d la senoide siga la envolvente del audio og
0.2 es escala global para q no sature
"""
sine = 0.2 * rms_up * torch.sin(phase)

"""
desconecta el tensor del grafo d gradientes, lo mueve a la cpu y añade dimension d canal para q torchaudio lo acepte como audio mono. [1,n_samples]
"""
out = sine.detach().cpu().unsqueeze(0)

torchaudio.save("recon_sine_loudness.wav", out, sr)

print("guardado: recon_sine_loudness.wav")
print("shape salida:", out.shape)
print("sample rate:", sr)
print("key:", key)