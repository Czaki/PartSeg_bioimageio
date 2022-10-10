# How to improve data labeling using deep learn on PartSeg output example

Deep learning could be used to improve data labeling, not only base on manually annotated data,
but also base on automatic segmentation from selected subset of images.

The example cases are:

1) Part of data are higher quality, that allow to label using simpler method and
   artificial noise and artifacts could be added as a preprocessing deep learn data.
2) Data contains channels that are not used by labeling algorithm, but are correlated with labeled objects.
   For example transmitted light, or unspecific bind of some probe.

The advantage of using labeling by algorithm over manual is that it is faster even if we
count time needed for verification of such segmentation.

In this tutorial we will us data from (here publication link) and processed using [Trapalyzer](https://github.com/Czaki/Trapalyzer)

## Data description

Data contains 2D images with 2 channels from fluorescence microscope and 3 channels(RGB) that contain information
from transmitted light microscope.
The Trapalyzer use only 2 channels from fluorescence microscope to perform segmentation so is fragile to artifacts
connected with background auto fluorescence that are marked as "Unknown extra" class.

## Steps

### Data filtering

Base on manual verification and results of measurements, part of data should be moved to verification set (different from train and test).
The labeling of elements of this set should be verified manually.


Sample code available https://github.com/Czaki/entropy-calculation/
