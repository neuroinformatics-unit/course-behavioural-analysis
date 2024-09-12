# %%
# From
# https://colab.research.google.com/github/dattalab/keypoint-moseq/blob/main/docs/keypoint_moseq_colab.ipynb

from pathlib import Path
%matplotlib widget

# %%
import keypoint_moseq as kpms
import matplotlib.pyplot as plt
import numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data - if SLEAP, it needs to be SLEAP H5 FILE
# data_dir='/home/sminano/swc/project_teaching_behaviour/mouse-EPM/derivatives/behav/software-SLEAP_project/predictions/video-1.predictions.slp'
# sleap_file='video-1.predictions.slp'
# sleap_file = os.path.join(data_dir, sleap_file)
# sleap_predictions = '/home/sminano/swc/project_teaching_behaviour/mouse-EPM/derivatives/behav/software-SLEAP_project/predictions/video-1.predictions.slp'
video_path = (
    Path().cwd().parents[1]
    / "mouse-EPM"
    / "rawdata"
    / "sub-01_id-M708149"
    / "ses-01_date-20200317"
    / "behav"
    / "video-1.mp4"  # sub-01_ses-01_task-EPM_time-165049_
)
sleap_predictions = (
    Path().cwd().parents[1]
    / "mouse-EPM"
    / "derivatives"
    / "behav"
    / "software-SLEAP_project"
    / "predictions"
    / "video-1.predictions.analysis.h5"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create moseq project
project_dir = "/home/sminano/swc/project_teaching_behaviour/mouse-EPM-moseq-video-1"  # must not exist before
config = lambda: kpms.load_config(project_dir)

# use SLEAP predictions file
kpms.setup_project(
    project_dir, sleap_file=sleap_predictions
)  # this creates a config file

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Edit config
kpms.update_config(
    project_dir,
    video_dir=str(Path(video_path).parent),
    anterior_bodyparts=["snout", "left_ear", "right_ear"],  # used to initialize heading
    posterior_bodyparts=["tail_base"],  # used to initialize heading
    # use_bodyparts= # determines the subset of bodyparts to use for modeling and the order in which they are represented
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parse keypoint data & load
coordinates, confidences, bodyparts = kpms.load_keypoints(
    str(sleap_predictions), "sleap"
)

# - coordinates: for each video, an array of size (nframes, n_kpts, n_spatial_dims) with the coords of the keypoints in image coord system
# - confidences: for each video, an array of size (nframes, n_kpts)
print(coordinates.keys())
print(coordinates["video-1.predictions.analysis"].shape)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot coordinates
# import matplotlib.pyplot as plt

# coords = coordinates['video-1.predictions.analysis']

# # trajectory of first kpt in time
# fig, ax = plt.subplots(1,1)
# im = ax.scatter(
#     x=coords[:,0,0],  # first kpt
#     y=coords[:,0,1],  # first kpt
#     s=1,
#     c=range(coords.shape[0]),
#     vmin=0,
#     vmax=coords.shape[0]
# )

# # Add a colorbar based on the scatter plot
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label('frame')  # Optional: label for the colorbar
# ax.set_aspect("equal")
# ax.set_xlabel("x (pixels)")
# ax.set_ylabel("y (pixels)")
# ax.invert_yaxis()


# #### color by confidence

# # trajectory of first kpt in time
# fig, ax = plt.subplots(1,1)
# im = ax.scatter(
#     x=coords[:,0,0],  # first kpt
#     y=coords[:,0,1],  # first kpt
#     s=1,
#     c=confidences['video-1.predictions.analysis'][:,0], # first kpt
#     vmin=np.nanmin(confidences['video-1.predictions.analysis'][:,0]),
#     vmax=np.nanmax(confidences['video-1.predictions.analysis'][:,0])
# )

# # Add a colorbar based on the scatter plot
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label('confidence')  # Optional: label for the colorbar
# ax.set_aspect("equal")
# ax.set_xlabel("x (pixels)")
# ax.set_ylabel("y (pixels)")
# ax.invert_yaxis()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Format data for modeling
data, metadata = kpms.format_data(coordinates, confidences, **config())

print(data.keys())
print(metadata)  # not sure what this is

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Confidence calibration regression --- skipped

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fit PCA
pca = kpms.fit_pca(**data, **config())

kpms.save_pca(pca, project_dir)  # saves it to the project

# visualise results
kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())

# save number of dimensions to keep
kpms.update_config(project_dir, latent_dim=4)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Overview model fitting
# 1. Random initialisation
# 2. Fit an AR-HMM
# 3. Fit full model
# 4. Extract model results

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Setting kappa
# Most users will need to adjust the kappa hyperparameter to achieve
# the desired distribution of syllable durations.

# optionally modify kappa
# model = kpms.update_hypparams(model, kappa=NUMBER)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 1. Randomly initialize the model
model = kpms.init_model(data, pca=pca, **config())

print("Model keys")
print(model.keys())

print("Model hparams")
print(model["hypparams"])

print("Model states")
print(model["states"].keys())

# ---> dict_keys(['z', 'x', 'v', 'h', 's'])
# - syllable labels (z)
# - inferred low-dim pose state (x)
# - inferred centroid (v)
# - inferred heading (h)
#     ⋮
# ```


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2. Fit AR-HMM
num_ar_iters = 100

model, model_name = kpms.fit_model(
    model, data, metadata, project_dir, ar_only=True, num_iters=num_ar_iters
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3. Fit full model
# You may need to try a few values of kappa

# load model checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir, model_name, iteration=num_ar_iters
)

# modify kappa to maintain the desired syllable time-scale
# initial kappa: 1000000.0
# see reference kappa values for the paper datasets in supplementary table
model = kpms.update_hypparams(model, kappa=1e4)

# run fitting for an additional 500 iters
model = kpms.fit_model(
    model,
    data,
    metadata,
    project_dir,
    model_name,
    ar_only=False,
    start_iter=current_iter,  # start where you left off!
    num_iters=current_iter + 500,
)[0]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 4. Extract model results
# Parse the modeling results and save them to `{project_dir}/{model_name}/results.h5`.
# The results are stored as follows, and can be reloaded at a later time using `kpms.load_results`.
# Check the docs for an [in-depth explanation of the modeling results](https://keypoint-moseq.readthedocs.io/en/latest/FAQs.html#interpreting-model-outputs).
# ```
#     results.h5
#     ├──recording_name1
#     │  ├──syllable      # syllable labels (z)
#     │  ├──latent_state  # inferred low-dim pose state (x)
#     │  ├──centroid      # inferred centroid (v)
#     │  └──heading       # inferred heading (h)
#     ⋮
# ```

# modify a saved checkpoint so that syllables are ordered by frequency
kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

# load the most recent checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)

# extract results to variable and h5 file
results = kpms.extract_results(model, metadata, project_dir, model_name)
# to read: load_results(path=path-to-h5) --> dict

# optionally save results as csv
kpms.save_results_as_csv(results, project_dir, model_name)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inspect results

print(results["video-1.predictions.analysis"].keys())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Count freq of syllables across all video
syllables_per_frame = results["video-1.predictions.analysis"][
    "syllable"
]  # array: (nframes, )
syllables_count = {}
for syl in np.unique(syllables_per_frame):
    syllables_count[syl] = sum(syl == syllables_per_frame)

# sort by count
syllables_count = dict(
    sorted(syllables_count.items(), key=lambda item: item[1], reverse=True)
)

# print top 10
# (they should be sorted by frequency?)
n_frames = results["video-1.predictions.analysis"]["syllable"].shape[0]
for syl, count in list(syllables_count.items())[:10]:
    print(f"Syllable id-{syl}: {(count/n_frames)*100:.2f} % of frames")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot ethogram for first video

import itertools

# find lengths of continuous tracks
# itertools.groupby: generates a break or new group
# every time the value of the key function changes
syllable_chunks = [
    (key, len(list(group_iter)))
    for key, group_iter in itertools.groupby(syllables_per_frame)
]  # list of (syllable_id, len)

list_durations = [dur for syl, dur in syllable_chunks]
start_chunks = np.cumsum([0]+list_durations) - 0.5

list_colors = (
    plt.get_cmap("tab10").colors
    + plt.get_cmap("tab20b").colors
    + plt.get_cmap("Set3").colors
    + plt.get_cmap("Set1").colors
)  # 51

frames_max_to_plot = 1000

fig, ax = plt.subplots(1, 1, figsize=(8,5))
rects = ax.barh(
    y=results.keys(),
    width=[syl_dur for syl_id, syl_dur in syllable_chunks],
    left=start_chunks[:-1],  # starting frame of each chunk - 0.5
    height=1,
    color=[
        list_colors[syl_id%len(list_colors)] 
        for syl_id, syl_dur in syllable_chunks
    ],
)
ax.bar_label(
    rects, 
    labels=[syl_id for syl_id, syl_dur in syllable_chunks],
    label_type='center', 
    color='white'
)
ax.set_xlim(0, frames_max_to_plot)
ax.set_xlabel('frames')
ax.yaxis.set_visible(False)

ax.set_aspect(100)
ax.set_title(*results.keys())


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Count median duration per syllable

# compute median syllable duration
median_syllable_duration = np.median([syl_dur for (syl_id, syl_dur) in syllable_chunks])

# compute median duration per syllable ID
median_duration_per_syl = {}
for syl in list(syllables_count.keys()):
    median_duration_per_syl[syl] = np.median(
        [syl_dur for (syl_id, syl_dur) in syllable_chunks if syl_id == syl]
    )  # frames


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot median duration per syllable
fps = 30  # fps

fig, ax = plt.subplots(1, 1)
ax.scatter(
    x=median_duration_per_syl.keys(),
    y=median_duration_per_syl.values(),
)
ax.hlines(
    y=median_syllable_duration,
    xmin=-1,
    xmax=len(median_duration_per_syl) + 1,
    colors="r",
)
ax.set_xlabel("syllable ID")
ax.set_ylabel("median duration (frames)")

print(f"Median syllable duration (frames): {median_syllable_duration}")
print(f"Median syllable duration (ms): {1000*median_syllable_duration/fps}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise most frequent syllable
frames_max_to_plot = len(syllables_per_frame)
list_syllables_to_plot = [1,3,2, 4, 42, 31]
for selected_syl in list_syllables_to_plot:

    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    rects = ax.barh(
        y=results.keys(),
        width=[syl_dur for syl_id, syl_dur in syllable_chunks],
        left=start_chunks[:-1],  # starting frame of each chunk
        height=1,
        color=[
            'red' if syl_id==selected_syl 
            else 'grey' 
            for syl_id, _ in syllable_chunks
        ],
    )
    # ax.bar_label(rects, label_type='center', color='white')
    ax.set_xlim(0, frames_max_to_plot)
    ax.set_xlabel('frames')
    ax.yaxis.set_visible(False)

    ax.set_aspect(int(frames_max_to_plot/10))
    ax.set_title(f'{list(results.keys())[0]} - syllable {selected_syl}')
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Trajectory plots per syllable
results = kpms.load_results(project_dir, model_name)
kpms.generate_trajectory_plots(
    coordinates, results, project_dir, model_name, **config()
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Other viz:

# Grid movies
# kpms.generate_grid_movies(results, project_dir, model_name, coordinates=coordinates, **config());

# Syllable dendrogram
# kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **config())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Apply to new data (inference)
## Apply to new data

# The code below shows how to apply a trained model to new data.
# This is useful if you have performed new experiments and would like
# to maintain an existing set of syllables. The results for the new
# experiments will be added to the existing `results.h5` file.

# load the most recent model checkpoint and pca object
model = kpms.load_checkpoint(project_dir, model_name)[0]

# load new data 
new_data = "/home/sminano/swc/project_teaching_behaviour/mouse-EPM/derivatives/behav/software-SLEAP_project/predictions/video-2.predictions.analysis.h5"  # can be a file, a directory, or a list of files
coordinates, confidences, bodyparts = kpms.load_keypoints(str(new_data), "sleap")

# format data for model
data, metadata = kpms.format_data(coordinates, confidences, **config())

# apply saved model to new data
results = kpms.apply_model(
    model, 
    data, 
    metadata, 
    project_dir, 
    model_name, 
    **config(), 
    results_path=(
        '/home/sminano/swc/project_teaching_behaviour/mouse-EPM-moseq-video-1/'
        '2024_09_12-18_32_02/results_video-2.h5'
    )
)  # ----> overwrites results.h5 file!

# optionally rerun `save_results_as_csv` to export the new results
kpms.save_results_as_csv(results, project_dir, model_name)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read new results

results_all = kpms.load_results(
    path='/home/sminano/swc/project_teaching_behaviour/mouse-EPM-moseq-video-1/2024_09_12-18_32_02/results.h5'
)
results.keys()

# %%
