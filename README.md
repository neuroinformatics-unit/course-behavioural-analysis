# Video-based analysis of animal behaviour
This repository hosts the slides and materials for the
[behavioural analysis course](https://software-skills.neuroinformatics.dev/courses/video-analysis.html) taking place annualy at the [Sainsbury Wellcome Centre (SWC)](https://www.sainsburywellcome.org/web/).

The rendered slides are available [here](https://neuroinformatics.dev/course-behavioural-analysis/#/title-slide).

## Overview
This is an introductory course on analysing animal behaviour from video data. The course covers:

- Overview of animal tracking methods and terminology.
- **Pose estimation and tracking with [SLEAP](https://sleap.ai/)**. This includes hands-on practice with training a pose estimation model, evaluating its performance, and using it to predict pose tracks in new videos.
- **Analysing pose tracks with [movement](https://movement.neuroinformatics.dev/)**. This includes loading the predicted pose tracks in Python, filtering and smoothing them, computing kinematic variables, and visualising the results.
- **Extracting behavioural syllables with [keypoint-moseq](https://keypoint-moseq.readthedocs.io/en/latest/index.html)**.

## Instructors
* [Niko Sirmpilatze ](https://github.com/niksirbi)
* [Chang Huan Lo](https://github.com/lochhh)
* [Sofía Miñano](https://github.com/sfmig)

## Prerequisites

### Hardware Requirements

This is a hands-on course, so **please bring your own laptop and charger**. A mouse is recommended but not essential. A dedicated GPU is not required but will be helpful.

### General Software Requirements

> [!Note]
> If you are attending the entire [General Software Skills for Systems Neuroscience](https://software-skills.neuroinformatics.dev/courses/software-skills.html) course, you will have already installed the general software requirements during Day 1 and may skip this section.

- An IDE for Python programming. We recommend one of the following:
  - [Visual Studio Code](https://code.visualstudio.com/) with the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
  - [PyCharm](https://www.jetbrains.com/pycharm/)
  - [JupyterLab](https://jupyter.org/install)

- A working `conda` (or `mamba`) installation. If you don't have it, install via [Miniforge](https://github.com/conda-forge/miniforge).
- A working [Git](https://git-scm.com/) installation.

### Specific Software Requirements

> [!Note]
> Only proceed with these installations after completing the above general software requirements.

You will need to pre-install two different `conda` environments for the practical exercises. Create them as follows:

1. [**SLEAP**](https://sleap.ai/): Use the [conda package method](https://sleap.ai/installation.html#conda-package) from the SLEAP installation guide. You may use `conda` instead of `mamba` in the installation command. If your machine lacks an NVIDIA GPU, that's fine; for this course, you just need to be able to launch the SLEAP GUI using `sleap-label`.
2. [**Keypoint-MoSeq**](https://keypoint-moseq.readthedocs.io): Use the recommended [conda installation method](https://keypoint-moseq.readthedocs.io/en/latest/install.html#install-using-conda).

You should now have two new conda environments called `sleap` and `keypoint_moseq`. To view all your conda environments, run `conda env list`.

### Sample Data

Download the sample data for this course from [Dropbox](https://www.dropbox.com/scl/fo/ey7b6yrqax2olqyv1th7j/h?rlkey=u4wh2gxtbbn4g5o3s55zbx6pp&st=zolupk4i&dl=0). Click "Download" to get the `behav-analysis-course.zip` archive and unzip it.

Alternatively, if you have access to the SWC's `ceph` filesystem, find the dataset at `/ceph/scratch/neuroinformatics-dropoff/behav-analysis-course`. To mount `ceph` on your laptop, follow the [instructions on the SWC wiki](https://wiki.ucl.ac.uk/display/SSC/Storage%3A+Ceph). Note: You must be connected to the SWC network to access the wiki and mount `ceph`.

Ensure you copy the data to a convenient location on your laptop.
