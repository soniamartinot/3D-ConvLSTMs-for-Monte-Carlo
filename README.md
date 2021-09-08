# 3D-ConvLSTMs-for-Monte-Carlo

This is the official repository for the article [***High-particle simulation of Monte-Carlo dose distribution with 3D ConvLSTMs***](https://github.com/soniamartinot/3D-ConvLSTMs-for-Monte-Carlo/blob/master/miccai_paper.pdf) presented in MICCAI 2021 (Strasbourg).

![](https://github.com/soniamartinot/3D-ConvLSTMs-for-Monte-Carlo/blob/master/case_3339.gif)
![](https://github.com/soniamartinot/3D-ConvLSTMs-for-Monte-Carlo/blob/master/case_3115.gif)

## Dependencies

```
- numpy==1.17.4
- torchvision==0.9.1+cu111
- tqdm==4.50.0
- torch==1.8.1+cu111
- matplotlib==3.3.2
- asposestorage==1.0.2
- pytorch_ssim==0.1
- SimpleITK==2.1.0
```

## Data

The Monte-Carlo dataset that we created and used for this research:
- is publically available [here](https://hosting.therapanacea.eu/data/miccai2021/), 
- has a comprehensive description [here](https://github.com/soniamartinot/3D-ConvLSTMs-for-Monte-Carlo/blob/master/readme_monte_carlo_vmat_dataset.pdf).

## Training 

To train a model, run `python train.py` in the folder where you wish to save training results. 

You can change all the training parameters by adding specifications. For example `python train.py --gpu 3` changes the identification number of your GPU to 3. All parameters can be found in the `parse_args()` function in `utils.py`.


You can find the weights of our best performing model (architecture: stacked 3D ConvLSTMs) from the article saved as `best_val_model.pt`.

## Citation

If you find our project useful, please cite:

```
@inproceedings{martinot2021convlstm,
    title={High-particle simulation of Monte-Carlo dose distribution with 3D ConvLSTMs},
    author={Martinot, Sonia and Bus, Norbert and Vakalopoulou, Maria and Robert, Charlotte and Deutsch, Eric and Paragios, Nikos},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
    year={2021},
    organization={Springer}
}
```


