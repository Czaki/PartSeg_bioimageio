# How to use bioimageio model in PartSeg

In this tutorial we show how to use deep learning models in *BioImage.IO* compatible format from *PartSeg* GUI using *PartSeg_bioimageio* plugin.

Such model need to be downloaded and user need to know on which channels are required by model. Typically this requires dowloading and umpacing the model,
loading the input data in PartSeg, selecting chanels that are required by model and executing.



## Model selction and installation

Visit [BioImage.IO](https://bioimage.io/) and download model that you would like to test on your data. In this tutorial we use our [model](put_link) for segmentation and classification of neutrofile that is going to be submitted to BioImage.IO soon. The downloaded moded should be extracted ito directory of your choosing.

## Preparation of data for processing with the model

In this tutorial we use a single image from [dataset](https://zenodo.org/record/7335430) published on Zenodo.

To load the data in PartSeg the user could drag and drop the file to PartSeg window or use *Open* button.

![Open file view](images/open_file_tr.png)


## Executing the model in PartSeg

On the top of the right panel select Bioimageio multilabel method:

![Bioimageio multilabel selection](images/select_model_tr.png)

Use "Select file" button end point to `rdf.yaml` in the location where the model has been extracted.
After PartSeg parses model parameters, based on model description, select which channel from image constains data expected by the model.

You could also adjust "Reconstruction Parameters" to remove artificial objects recognized by the model.

The *Background threshold* specifies minimum confidence level for classification of every pixel. Values bellow this threshold are treated as background.

*Minimum size* is responsible for filtering out objects smaller than given size in pixels.


![set basic model parameters](images/methd_params_adjust.png)

After channels are selected click **Execute** button and wait for result.

![result of model execute](images/model_output.png).

The information how to convert these steps to batch processing can be found in PartSeg official [documentation](https://partseg.readthedocs.io/en/latest/interface-overview/interface-overview.html#batch-processing).
