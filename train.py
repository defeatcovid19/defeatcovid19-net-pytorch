import copy
import diagnostics
from pathlib import Path
from datasets import ChestXRayPneumoniaDataset, COVIDChestXRayDataset
from models import Resnet34
from trainer import Trainer
from sklearn.model_selection import train_test_split, StratifiedKFold

batch_size = 64
size = 256
n_splits = 5

# Pretrain with Chest XRay Pneumonia dataset (>5k images)
pneumonia_classifier = Resnet34()
dataset = ChestXRayPneumoniaDataset(Path('input/chest-xray-pneumonia'), size)
train_idx, validation_idx = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    stratify=dataset.labels
)
trainer = Trainer(pneumonia_classifier, dataset, batch_size, train_idx, validation_idx)
trainer.run(max_epochs=2)

# Fine tune with COVID-19 Chest XRay dataset (~120 images)
dataset = COVIDChestXRayDataset(Path('input/covid_chestxray'), size)
print('Executing a {}-fold cross validation'.format(n_splits))
split = 1
skf = StratifiedKFold(n_splits=n_splits)
for train_idx, validation_idx in skf.split(dataset.df, dataset.labels):
    print('===Split #{}==='.format(split))
    # Start from the pneumonia classifier
    classifier = copy.deepcopy(pneumonia_classifier)
    trainer = Trainer(classifier, dataset, batch_size, train_idx, validation_idx)
    trainer.run(max_epochs=15)
    split += 1
