# How to improve data labeling using deep learn on PartSeg output example

## Plan

1) Tutorial opisuje jak stworzyć model aby poprawić segmentację wykonywaną przez PartSeg.
2) Jest on dla bioinformatyka, który nie chce wchodzić w detale danych i wykorzystać do tego output PartSeg-a.
   Jako wsparcie dla biologa który przygotował segmentację.
3) Jako wejście używamy zapisanych projektów PartSega, które zawierają segmentację i dane wejściowe.
4) Możliwe wartianty do uczenia gdzie może to pomóc:

   1) występują artefakty, które uniwmożliwiają klasyczną metodę segmentacji
   2) danę użyte do klasycznej segmentacji sa po dekonwolucji (lub innym odszumianiu), którego chce się uniknąć
   3) jest potrzba aby algorytm działał na bardziej zaszumionych danych

5) dodanie funkcji czytającej dane.

______________________________________________________________________


Deep learning could be used to improve data labeling, not only base on manually annotated data,
but also base on automatic segmentation from selected subset of images.

The example cases are:

1) Part of the data are higher quality, that allow to label using simpler method and
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

Based on manual verification and results of measurements, part of data should be moved to verification set (different from train and test).
The labeling of elements from this set should be verified manually.


### Model fit

For train model we use a nice package [torch-em](https://github.com/constantinpape/torch-em) [1].

First wee need to implement custom data reader to handle PartSeg projects [wkleić kod]



Sample code available https://github.com/Czaki/entropy-calculation/


## Citations

1. Constantin Pape, Fynn Beuttenmüller, JonasHell, & Wei Ouyang. (2022). constantinpape/torch-em: Jupyter notebooks (0.4.0). Zenodo. https://doi.org/10.5281/zenodo.6415314
