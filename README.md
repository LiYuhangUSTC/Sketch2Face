# Sketch2Face


This is the official implementation of paper **DeepFacePencil: Creating Face Images from Freehand Sketches**. [arXiv](https://arxiv.org/abs/2008.13343), [Project](https://liyuhangustc.github.io/Sketch2Face/)

Also see another project: [Lines2Face project](https://liyuhangustc.github.io/Lines2Face/)

![architecture](figure/architecture.png "architecture")

### Prerequisites
PyTorch 1.6, Python 3.7, NumPy, scipy, PIL, tqdm

### Data
We use [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) to obtain synthesized (boundary) sketches. Deformed sketches are generated by vectorizing synthesized sketches using [AutoTrace](http://autotrace.sourceforge.net/) and adding random offsets to endpoints and control points of vectorized strokes.

### Pretrained model
[GoogleDrive](https://drive.google.com/drive/folders/1Oga57eTNEIbuxV6L9GxGYv9fyIEdcBCD?usp=sharing)

### Test
The test sketches should be put in folder `./datasets/CelebAMask/test_A`. The pretrained model should be put in folder `./checkpoints/pretrained`. Example script of testing can be found in `./test_scripts.sh`. The results are supposed to be in `./results`. The deform.pth file is uploaded to Google Drive.

### Cite
```
@inproceedings{DeepFacePencil,
 author = {Li, Yuhang and Chen, Xuejin and Yang, Binxin and Chen, Zihan and Cheng, Zhihua and Zha, Zheng-Jun},
 title = {DeepFacePencil: Creating Face Images from Freehand Sketches},
 booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
 series = {MM '20},
 year = {2020},
 isbn = {978-1-4503-7988-5/20/10},
 location = {Seattle, WA, USA},
 pages = {},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/3394171.3413684},
 doi = {10.1145/3394171.3413684},
 acmid = {3413684},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Sketch-based synthesis, face image generation, spatial attention, dual generator, conditional generative adversarial networks},
} 
```

### Credits
This code borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
