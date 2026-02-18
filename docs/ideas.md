# Ideas sobre lo que hacer el TFG

añadirle inputs 

en vez de ser escritos, que sean con variables flow sliders

lo mismo hay una manera de usar midi para controlar los parametros
aunque creo que eso era lo del ddsp

tmb se le podría pasar audio y que lo convierta en el instrumento deseado

una vez generado un sonido, tener una interfaz estilo arbol o plano para poder explorar otros sonidos cercanos en su espacio latente

Puesto que tenemos labels comunes para los datasets de entrenamiento, podriamos hacer que la difusion fuera condicionada
podríamos probar un modelo de difución 1d para generar audio directamente a ver que pasa

## Documentos interesantes
* [A survey of deep learning audio generation methods](https://arxiv.org/pdf/2406.00146):

* [MULTI-INSTRUMENT MUSIC SYNTHESIS WITH SPECTROGRAM DIFFUSION](https://arxiv.org/pdf/2206.05408)
    [github](https://github.com/magenta/music-spectrogram-diffusion)
    [github de otra implementacion](https://github.com/yoyolicoris/music-spectrogram-diffusion-pytorch) -> también tiene datasets interesantes
    -> Para generar cada chunk de audio (5s) emplean el chunk anterior a modo de contexto para que las transiciones sean suaves, puede ser una solución a lo que comentamos de cortes

* [ejemplo difusión](https://towardsdatascience.com/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946/)
* [ejemplo difusión](https://medium.com/@meghavalgi/how-to-build-a-diffusion-model-from-scratch-in-python-complete-guide-4f7793c3c711)

* [más difusión](https://medium.com/@meghavalgi/bringing-text-to-life-fine-tuning-stable-diffusion-02ed7c75a83f)

* [DALL-E 3: Latent diffusion in compressed spaces for efficiency.](https://medium.com/@kdk199604/latent-diffusion-model-efficient-high-resolution-image-synthesis-without-compromise-1bec1bee5f8b)

* [Paper de lo anterior](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)

* [Github de lo anterior](https://github.com/CompVis/latent-diffusion)

* [VQ-VAE](https://proceedings.neurips.cc/paper_files/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)

* https://medium.com/data-science/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946
