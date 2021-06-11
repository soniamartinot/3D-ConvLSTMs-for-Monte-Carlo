# 3D-ConvLSTMs-for-Monte-Carlo

This is the official repository for the article ***High-particle simulation of Monte-Carlo dose distribution with 3D ConvLSTMs*** presented in MICCAI 2021 (Strasbourg).

![](https://github.com/soniamartinot/3D-ConvLSTMs-for-Monte-Carlo/blob/master/case_3339.gif)
![](https://github.com/soniamartinot/3D-ConvLSTMs-for-Monte-Carlo/blob/master/case_3115.gif)


To train a model, run `python train.py` in the folder where you wish to save training results. 

You can change all the training parameters by adding specifications. For example `python train.py --gpu 3` changes the identification number of your GPU to 3. All parameters can be found in the `parse_args()` function in `utils.py`.


You can find the weights of our best performing model (architecture: stacked 3D ConvLSTMs) from the article saved as `best_val_model.pt`.


The dataset used for this article will be made available online shortly.
