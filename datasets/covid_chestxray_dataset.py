import cv2
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class COVIDChestXRayDataset(Dataset):
    def __init__(self, path, size=128, augment=None):
        super(COVIDChestXRayDataset, self).__init__()
        print('{} initialized with size={}, augment={}'.format(self.__class__.__name__, size, augment))
        print('Dataset is located in {}'.format(path))
        self.size = size
        self.augment = augment
        
        image_dir = path / 'images'
        metadata_path = path / 'metadata.csv'
        
        df_metadata = pd.read_csv(metadata_path, header=0)
        # Drop CT scans
        df_metadata = df_metadata[df_metadata['modality'] == 'X-ray']
        # Keep only PA/AP/AP Supine, drop Axial, L (lateral)
        allowed_views = ['PA', 'AP', 'AP Supine']
        df_metadata = df_metadata[df_metadata['view'].isin(allowed_views)]
        
        # COVID-19 = 1, SARS/ARDS/Pneumocystis/Streptococcus/No finding = 0
        self.labels = (df_metadata.finding == 'COVID-19').values.reshape(-1, 1)
        images = df_metadata.filename
        images = images.apply(lambda x: image_dir / x).values.reshape(-1, 1)
        
        self.df = pd.DataFrame(np.concatenate((images, self.labels), axis=1), columns=['image', 'label'])
        
        del images

        print("Dataset: {}".format(self.df))
            

    @staticmethod
    def _load_image(path, size):
        img = Image.open(path)
        img = cv2.resize(np.array(img), (size, size), interpolation=cv2.INTER_AREA)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
            img = np.dstack([img, img, img])
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # size, size, chan -> chan, size, size
        img = np.transpose(img, axes=[2, 0, 1])
        
        return img
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = self._load_image(row['image'], self.size)
        label = row['label']        

        if self.augment is not None:
            img = self.augment(img)

        return img, label

    def __len__(self):
        return self.df.shape[0]