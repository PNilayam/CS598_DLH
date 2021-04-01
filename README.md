# A Neural Way to Predict Length of Stay

Group project for CS 598 Deep learning for Healthcare : University of Illinois Urbana-Champaign under Prfessor Jimeng Sun

# GPU / CUDA setup

Follow instructions [here](https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1) 

Note that CUDA 11.1 was used which is currently compatible with the stable version of pytorch.

Use `watch -n 2 nvidia-smi` to see if the GPU is being used.

# Conda environment
Conda was used to create the development enviornment first creating a new conda environment

`conda create dl4h`

After activating the new environment install pytorch using conda install:

`conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge`


  
