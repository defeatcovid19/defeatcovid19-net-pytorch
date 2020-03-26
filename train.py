import diagnostics
from pathlib import Path
from datasets import ChestXRayPneumoniaDataset, COVIDChestXRayDataset
from models import Resnet34
from trainer import Trainer

batch_size = 64
size = 256
classifier = Resnet34()

dataset = ChestXRayPneumoniaDataset(Path('input/chest-xray-pneumonia'), size)

trainer = Trainer(classifier, dataset, batch_size)
trainer.run(max_epochs=2)

dataset = COVIDChestXRayDataset(Path('input/covid_chestxray'), size)

trainer = Trainer(classifier, dataset, batch_size)
trainer.run(max_epochs=15)
