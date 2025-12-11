# Ideas sobre lo que hacer el TFG

añadirle inputs 

en vez de ser escritos, que sean con variables flow sliders

lo mismo hay una manera de usar midi para controlar los parametros
aunque creo que eso era lo del ddsp

tmb se le podría pasar audio y que lo convierta en el instrumento deseado

una vez generado un sonido, tener una interfaz estilo arbol o plano para poder explorar otros sonidos cercanos en su espacio latente

podríamos probar un modelo de difución 1d para generar audio directamente a ver que pasa

## Documentos interesantes
* [A survey of deep learning audio generation methods](https://arxiv.org/pdf/2406.00146):

* [MULTI-INSTRUMENT MUSIC SYNTHESIS WITH SPECTROGRAM DIFFUSION](https://arxiv.org/pdf/2206.05408)
    [github](https://github.com/magenta/music-spectrogram-diffusion)
    [github de otra implementacion](https://github.com/yoyolicoris/music-spectrogram-diffusion-pytorch) -> también tiene datasets interesantes
    -> Para generar cada chunk de audio (5s) emplean el chunk anterior a modo de contexto para que las transiciones sean suaves, puede ser una solución a lo que comentamos de cortes

* https://medium.com/data-science/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946