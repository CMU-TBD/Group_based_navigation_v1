# Overview

This is the code used for the paper **Group-based Motion Prediction for Navigation in Crowded Environments**
https://proceedings.mlr.press/v164/wang22e.html

If you find this code useful, please cite
```

@InProceedings{pmlr-v164-wang22e,
  title = {Group-based Motion Prediction for Navigation in Crowded Environments},
  author = {Wang, Allan and Mavrogiannis, Christoforos and Steinfeld, Aaron},
  booktitle = {Proceedings of the 5th Conference on Robot Learning},
  pages = {871--882},
  year = {2022},
  editor = {Faust, Aleksandra and Hsu, David and Neumann, Gerhard},
  volume = {164},
  series = {Proceedings of Machine Learning Research},
  month = {08--11 Nov},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v164/wang22e.html},
}
```

# Note

The code in this repo is no longer actively maintained. It has been cleaned up and moved to https://github.com/allanwangliqian/dataset-crowd-simulation

# Prerequisites

Environment: Python3

In order to run our experiments, the following major packages are needed:
pickle, numpy, scipy, opencv, matplotlib, attrdict, PyTorch

A GPU with 4GB memory will greatly boost the performance for certain experiments.

Additionally, please refer to https://github.com/sybrenstuvel/Python-RVO2
to install the python-RVO2 package

# Training the autoencoders for our group prediction policies

To train the autoencoder models, run:
```
python train.py --dset <dset_num>
```
`<dset_num>` is the dataset to be evaluated on, where

* 0 - ETH
* 1 - HOTEL
* 2 - ZARA1
* 3 - ZARA2
* 4 - UNIV

With the specified `<dset_num>`, the model will train with data generated from the other 4 datasets.

For convenience, trained models can be found on https://drive.google.com/drive/u/2/folders/16tf6X2KECU6AZHub0UedeYAXmonu-Fnn

Please place them in the "checkpoints" folder.

Run `train.py` will overwrite the checkpoints stored here.

# Running the policies

To generate the results files with different policies, run:

```
python MPC.py
```
And follow the prompts to configure whether to run pedestrian-based or group-based policies,
whether to allow future state predictions, whether to enable reactive agents or whether to
enable simulated laser scans.

For convenience, the result files for all the policies are already pre-computed and stored in
the "results" folder. Run `MPC.py` will overwrite the result files stored here.

To bringup the evaluations of the result files, run:

```
python evaluate.py --metric <metric_num> --policy1 <policy_num> --policy2 <policy_num>
```
`--metric`, `--policy1` and `--policy2` flags are used to perform the Mann-Whitney u-test

`<metric_num>` specifies the metric to be tested on, where
* 0 - success rates
* 1 - minimum distance to pedestrians
* 2 - path lengths

`<policy_num>` specifies the policy to be picked for the statistical test, where

* 0 - ped-nopred
* 1 - ped-linear
* 2 - ped-sgan
* 3 - group_nopred
* 4 - group_auto
* 5 - laser-group-auto

After `evaluate.py` is run, you need to answer a single prompt specifying whether loading the results
run in reactive agent settings or non-reactive agent settings.

The displayed results follow the results in Table 3 and 4.
Afterwards, the results of the statistical test for the two chosen policies on the chosen metric
are shown. Each row of p-values from left to right correspond to evaluation in
ETH, HOTEL, ZARA1, ZARA2, UNIV datasets.

# Group intrusion tests

The above evaluation does not contain evaluation on group intrusions as this is done post-hoc.
After all the policies have been evaluated by `MPC.py`, run:

```
python group_intrusion_test.py
```
This will generate the result files to test for group intrusions. For convenience, the result files
are already pre-generated and are stored in "group_intrusion_rst". Running `group_intrusion_test.py`
will overwrite these result files.

To bring up the group intrusion evaluations, run:

```
python group_intrusion_interp.py --policy1 <policy_num> --policy2 <policy_num>
```
<policy_num> follows the same specification as above (when running evaluate.py).
A single prompt will again show up asking whether loading the results
run in reactive agent settings or non-reactive agent settings.
The displayed results also follow the group intrusion metric rows in Table 3 and 4, with
the results of the statistical test displayed afterwards.

# Reference
All the codes within the "sgan" folder are either directly used or slightly modified
to run our experiments.

They are developed by the Authors of SocialGAN
A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, and A. Alahi.
Social GAN: Socially acceptable324trajectories with generative adversarial networks.
InProceedings of the IEEE Conference on325Computer Vision and Pattern Recognition (CVPR),
pages 2255–2264, 2018

Author: Agrim Gupta
Link: https://github.com/agrimgupta92/sgan

