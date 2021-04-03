# A Neural Way to Predict Length of Stay

Group project for CS 598 Deep learning for Healthcare : University of Illinois Urbana-Champaign under Prfessor Jimeng Sun

# Environment Setup
## GPU / CUDA setup

Follow instructions [here](https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1) 

Note that CUDA 11.1 was used which is currently compatible with the stable version of pytorch.

Use `watch -n 2 nvidia-smi` to see if the GPU is being used.

## Conda environment (project)
Conda was used to create the development enviornment first creating a new conda environment

`conda create dl4h`

After activating the new environment install pytorch using conda install:

`conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge`


## Conda environment (benchmark)
The benchmark used for comparison requires an environment created with Keras/Tensorflow. The environment must include:
* Python 3.6
* Tensorflow
* Keras

Ensure that you create the environment according to requirements.txt

To run with GPU you will likely need to uninstall tensorflow and then re-install
`pip uninstall tensorflow`

This will also uninstall keras. Reinstall with:
`conda uninstall tensorflow-gpu'

Also re-install keras:
`conda install keras`

To change back to CPU you can uninstall tensorflow using conda and then re-install:
'conda uninstall tensorflow-gpu`
`conda install tensorflow`
`conda install keras`

 
