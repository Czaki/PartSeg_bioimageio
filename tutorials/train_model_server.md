# Guide To train model using GPU server

This is Guide how to train model using GPU server.
As there are various available server configuration you may need to adjust some of the steps.
We have access to GPU cluster managed using slurm and all scripts related to training steep
are stored in this [repository](https://github.com/Czaki/entropy-calculation/) `!! Przenieść do czegoś bez DVC skrypty`


## Data preparation

We describe here supervised training. So you need to prepare data with ground truth.
The possible options are:

1) Manual annotation (for example using napari)
2) Semi-automatic annotation (for example using napari with [`napari-segment-anything`](https://github.com/JoOkuma/napari-segment-anything) plugin)
3) Fully automatic annotation on low noise data* (for example using PartSeg).

## Data synchronization

When on project there is more than one person working on data quality it is important to
have procedures for data synchronisation. From our experience the usage of mail or mobile drives
often leads to de-synchronization. That leads to having obsolete train runs.

For smaller data one could use git-LFS, but for data size common for bio-imaging it is not possible for publicly available git providers.
We found that good alternative whit great integration with GIT is [DVC](https://dvc.org/).
In our workflow we used google drive as storage for data and DVC as synchronization tool.

## Data publication

The DVC is great for data synchronization but it is not good for data publication.
There is no good backend for DVC that promises long term data availability.

The good place to publish final data and trained model is [zenodo](https://zenodo.org/).
This is service hosted by CERN have long term funding that promises long term data availability.
Each dataset set is assigned DOI that allows to cite it in publications.


## Environment preparation

It is important to have reproducible environment for training.
If one use `pip` to install packages then should save version of installed packages using
`pip freeze > requirements.txt`, or create `requirements.in` file and use `pip-compile` to create `requirements.txt`.
Both way allow to recreate environment using `pip install -r requirements.txt` command.

When using conda it is possible to use `conda env export > environment.yml` to save environment and `conda env create -f environment.yml` to recreate it.

For other package managers and languages please check documentation.
If there is no such option it is required to manually write down all packages and versions used. Also indirect dependencies should be included.

## Training

We provide example scripts based on [`torch_em`](https://github.com/constantinpape/torch-em) library.

The [repository](https://github.com/Czaki/entropy-calculation/) contains 2 important scripts:

1) `my_proj_train.py` - the script to trains model
2) `sbatch_run.py` - the project to generate sbatch scripts and put jobs in queue
