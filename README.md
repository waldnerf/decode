# Scalable satellite-based delineation of field boundaries with DECODE

Official [mxnet](https://mxnet.incubator.apache.org/) implementation of the paper: ["Detect, consolidate, delineate: scalable mapping of field boundaries using satellite images"](https://www.mdpi.com/2072-4292/13/11/2197), Waldner et al. (2021). This repository contains source code for implementing and training the FracTAL ResUNet as described in the paper.  All models are built with the mxnet DL framework (version < 2.0), under the gluon api. We do not provide pre-trained weights. 

Inference examples for six areas in Australia.
![mantis](images/decode.png)


### Directory structure: 

```
.
├── examples
├── images
├── FracTAL_ResUNet
    │   ├── heads
    │   └── semanticsegmentation
    │       └── x_unet
    ├── nn
    │   ├── activations
    │   ├── layers
    │   ├── loss
    │   ├── pooling
    │   └── units
└── postprocessing
```

In  ```examples```, there are notebooks that 1) show how to initiate a Fractal ResUNet model, and perform forward and multitasking backward operations and 2) that illustrate how to perform instance segmantion using hierarchical watershed segmentations.


### License
CSIRO BSTD/MIT LICENSE

As a condition of this licence, you agree that where you make any adaptations, modifications, further developments, or additional features available to CSIRO or the public in connection with your access to the Software, you do so on the terms of the BSD 3-Clause Licence template, a copy available at: http://opensource.org/licenses/BSD-3-Clause.



### CITATION
If you find the contents of this repository useful for your research, please cite:
```

@Article{rs13112197,
AUTHOR = {Waldner, Franc\{c}ois and Diakogiannis, Foivos I. and Batchelor, Kathryn and Ciccotosto-Camp, Michael and Cooper-Williams, Elizabeth and Herrmann, Chris and Mata, Gonzalo and Toovey, Andrew},
TITLE = {Detect, Consolidate, Delineate: Scalable Mapping of Field Boundaries Using Satellite Images},
JOURNAL = {Remote Sensing},
VOLUME = {13},
YEAR = {2021},
NUMBER = {11},
ARTICLE-NUMBER = {2197},
URL = {https://www.mdpi.com/2072-4292/13/11/2197},
ISSN = {2072-4292},
ABSTRACT = {Digital agriculture services can greatly assist growers to monitor their fields and optimize their use throughout the growing season. Thus, knowing the exact location of fields and their boundaries is a prerequisite. Unlike property boundaries, which are recorded in local council or title records, field boundaries are not historically recorded. As a result, digital services currently ask their users to manually draw their field, which is time-consuming and creates disincentives. Here, we present a generalized method, hereafter referred to as DECODE (DEtect, COnsolidate, and DElinetate), that automatically extracts accurate field boundary data from satellite imagery using deep learning based on spatial, spectral, and temporal cues. We introduce a new convolutional neural network (FracTAL ResUNet) as well as two uncertainty metrics to characterize the confidence of the field detection and field delineation processes. We finally propose a new methodology to compare and summarize field-based accuracy metrics. To demonstrate the performance and scalability of our method, we extracted fields across the Australian grains zone with a pixel-based accuracy of 0.87 and a field-based accuracy of up to 0.88 depending on the metric. We also trained a model on data from South Africa instead of Australia and found it transferred well to unseen Australian landscapes. We conclude that the accuracy, scalability and transferability of DECODE shows that large-scale field boundary extraction based on deep learning has reached operational maturity. This opens the door to new agricultural services that provide routine, near-real time field-based analytics.},
DOI = {10.3390/rs13112197}
}




`
