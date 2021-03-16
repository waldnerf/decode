# Scalable satellite-based delineation of field boundaries with DECODE

Official [mxnet](https://mxnet.incubator.apache.org/) implementation of the paper: ["Detect, consolidate, delineate: scalable mapping of field boundaries using satellite images"](https://arxiv.org/abs/2009.02062), Waldner et al. (2021). This repository contains source code for implementing and training the FracTAL ResUNet as described in the paper.  All models are built with the mxnet DL framework (version < 2.0), under the gluon api. We do not provide pre-trained weights. 

Inference examples for six areas in Australia.
![mantis](images/decode.png)


### Directory structure: 

```
.
├── demo
├── images
├── models
│   ├── heads
│   └── semanticsegmentation
│       └── x_unet
├── nn
│   ├── activations
│   ├── layers
│   ├── loss
│   ├── pooling
│   └── units
├── src
└── postprocessing
```

In  ```demo```, there are notebooks that 1) show how to initiate a Fractal ResUNet model, and perform forward and multitasking backward operations and 2) that illustrate how to perform instance segmantion using hierarchical watershed segmentations.


### License
CSIRO BSTD/MIT LICENSE

As a condition of this licence, you agree that where you make any adaptations, modifications, further developments, or additional features available to CSIRO or the public in connection with your access to the Software, you do so on the terms of the BSD 3-Clause Licence template, a copy available at: http://opensource.org/licenses/BSD-3-Clause.



### CITATION
If you find the contents of this repository useful for your research, please cite:
```
@article{waldner2020scalable,
    title={Detect, consolidate, delineate: scalable mapping of field boundaries using satellite images},
    author={François Waldner and Foivos I. Diakogiannis and Kathryn Batchelor and Michael Ciccotosto-Camp and Elizabeth Cooper Williams and Chris Herrmann and Gonzalo Mata and Andrew Toovey},
    year={2021}
}
`
