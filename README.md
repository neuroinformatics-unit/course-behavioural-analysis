# Video-based analysis of animal behaviour
This repository hosts the slides and materials for the
[behavioural analysis course](https://software-skills.neuroinformatics.dev/courses/video-analysis.html) taking place at the [Sainsbury Wellcome Centre (SWC)](https://www.sainsburywellcome.org/web/)
in November 2023.

The rendered slides are available [here](https://neuroinformatics.dev/course-behavioural-analysis/#/title-slide).

## Overview
This is an introductory course on analysing animal behaviour from video data. The course covers:

- Motivation
- Overview of animal tracking methods and terminology
- Pose estimation and tracking
  - Overview of existing tools
  - Labeling animal body parts
  - Training a model
  - Predicting poses
  - Evaluating performance
- Analysing pose tracks with Python
  - Loading and saving data
  - Filtering/smoothing
  - Visualising tracks
  - Time spent in regions of interest

## Prerequisites
Make sure to follow these steps before the course starts. If you encounter any issues, please contact Niko Sirmpilatze via the SWC slack.

### Set up your laptop with an IDE and conda
This is a hands-on course, so make sure to bring your **own laptop and charger**.
A dedicated GPU may make some exercises much more enjoyable, but is **not required**.

Make sure to have an IDE installed, for example:
- [Visual Studio Code](https://code.visualstudio.com/)
- [PyCharm](https://www.jetbrains.com/pycharm/)
- [Jupyter Lab](https://jupyter.org/install)

If you don't already have `conda` set up on your computer, 
you can do so via [miniconda](https://docs.conda.io/projects/miniconda/en/latest/).

### Install [SLEAP](https://sleap.ai) in a conda environment
We recommend reading the official [SLEAP installation guide](https://sleap.ai/installation.html).
If you already have `conda` installed, you may skip the `mamba` installation steps outlined there.
Instead, install the `libmamba-solver` for `conda`:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```
This will get you the much faster dependency resolution that `mamba` provides, without having to install `mamba` itself.
From `conda` version 23.10 onwards (released in November 2023), `libmamba-solver` [is anyway the default](https://conda.org/blog/2023-11-06-conda-23-10-0-release/).

After that, you can follow the [rest of the SLEAP installation guide](https://sleap.ai/installation.html#conda-package), substituting `conda` for `mamba` in the relevant commands.

**Windows and Linux**
```bash
conda create -y -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap=1.3.1
```

**MacOS X (including Apple Silicon)**
```bash
conda create -y -n sleap -c conda-forge -c anaconda -c sleap sleap=1.3.1
```
If your machine doesn't include a GPU, you may ignore the [GPU support](https://sleap.ai/installation.html#gpu-support) section of the installation instructions.

__Note__
We have chosen `v1.3.1` to ensure compatibility with the version of SLEAP installed on the SWC's HPC cluster, but more recent versions - e.g. `v1.3.3` - should work just as well.

### Get the sample data
You can download the sample data for this course from 
[DropBox](https://www.dropbox.com/scl/fo/ey7b6yrqax2olqyv1th7j/h?rlkey=u4wh2gxtbbn4g5o3s55zbx6pp&st=zolupk4i&dl=0).
Please click on "Download" to get the entire `behav-analysis-course.zip` archive, and unzip it.

Alternatively, if you have access to the SWC's `ceph` filesystem, you can find the dataset in `/ceph/scratch/neuroinformatics-dropoff/behav-analysis-course`. To mount `ceph` on your laptop, follow the [relevant instructions on the SWC wiki](https://wiki.ucl.ac.uk/display/SSC/Storage%3A+Ceph). You need to be connected to the SWC network both to read the wiki and to mount `ceph`.

No matter how you got the data, make sure to copy it to a convenient location on your laptop.

### Optional: bring your own data
Optionally, you may bring along any animal videos that you would like to analyse. This could simply be a video of your pet, or of the fox you spotted in your garden last night. Preferably, the videos should be in either `.mp4` or `.avi` format.
