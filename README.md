# defeatcovid19-net-pytorch

This repo provides a Pytorch solution for predictions on X-ray images for COVID-19 patients. 

## Motivation
It is intended to be used as a template for **defeatcovid19** group partecipants who like to contribute. You can find more info on our group's effort [here](https://github.com/defeatcovid19/defeatcovid19-project). At the moment we're actively trying to contact local hospitals to collect radiologic (mainly XRay and Eco) images to build a robust dataset for deep learning training.

## Implementation

The network of choice is ResNet34, provided by torchvision and pretrained on Imagenet. 
The net is first trained on the [Kaggle Chest X-Ray Pneumonia dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) (5856 images) and then on the [COVID-19 Chest X-Ray dataset](https://github.com/ieee8023/covid-chestxray-dataset) (123 usable images). 

Axial and lateral images were removed from the latter dataset. COVID-19 diagnoses were labelled 1, 0 otherwise (SARS/ARDS/Pneumocystis/Streptococcus/No finding).

### Requirements
An `environment.yml` file is provided to list the package requirements (mainly numpy, pandas, opencv, torch). The train entrypoint expects to find the aforementioned datasets in `./input`. Adjust your paths accordingly.


### Training
You can train the network and see the results of the cross validation with
```
python train.py
```

## Running with Docker

### Requirements
NVIDIA Driver Installation
[Docker installation](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
[NVIDIA Docker installation](https://github.com/NVIDIA/nvidia-docker)

### Build docker image
From the root of the repository (the image takes several minutes to build, due to download and compilation):
```
source tools/docker/setup.sh
```
Or if you are using shell fish:
```
source tools/docker/setup.fish
```
For running the training process:
```
dkrun train.py
```
## Results (initial)
The first part of the training (on the "Pneumonia" dataset) uses a simple 80/20 train/valid split. It achieves a ROC AUC score close to 1 for the selected fold. 
The second part of the training (on the "COVID" dataset) uses a more robust 5-fold cross validation and it results in a ~0.77 ROC AUC score.


## Citations
- Paul Mooney, Chest X-Ray Images (Pneumonia), Kaggle dataset, https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia, 2018
- Joseph Paul Cohen, COVID-19 image data collection, https://github.com/ieee8023/covid-chestxray-dataset, 2020

## License

This repo serves as a template for future effort of the **defeatcovid19** group and as such is intended to be released under the MIT license.
