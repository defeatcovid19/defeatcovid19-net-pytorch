import cv2
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class NIHCX38Dataset(Dataset):
    ''' Integrates the National Institutes of Health Clinical Center chest x-ray dataset.
    
    Dataset description: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
    Download the dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC to the folder input/nih-cx38
    Extract all image_??.tar.gz to the input/nih-cx38/images/ folder and ensure input/nih-cx38/Data_Entry_2017.csv is present.
    '''

    def __init__(self, path, size=128, augment=None, balance=False):
        super(NIHCX38Dataset, self).__init__()
        print('{} initialized with size={}, augment={}'.format(self.__class__.__name__, size, augment))
        print('Dataset is located in {}'.format(path))
        self.size = size
        self.augment = augment
        
        image_dir = path / 'images'
        metadata_path = path / 'Data_Entry_2017.csv'
        
        df_metadata = pd.read_csv(metadata_path)
        df_metadata['labels'] = df_metadata['Finding Labels'].str.split('|')
                
        # Pneumonia = 1, no Pneumonia = 0
        finding_mask = lambda df, finding: df['labels'].apply(lambda l: finding in l)
        pneumonia_mask = finding_mask(df_metadata, 'Pneumonia')
        
        if balance:
            pneumonia_indices = np.arange(len(pneumonia_mask))[pneumonia_mask]
            normal_indices = np.arange(len(pneumonia_mask))[~pneumonia_mask]
            if len(pneumonia_indices) < len(normal_indices):
                normal_indices = np.random.choice(normal_indices, len(pneumonia_indices))
            else:
                pneumonia_indices = np.random.choice(pneumonia_indices, len(normal_indices))
            self.labels = np.concatenate((
                np.zeros(len(normal_indices)),
                np.ones(len(pneumonia_indices))
            )).reshape(-1, 1)
            images = df_metadata['Image Index'][np.concatenate((normal_indices, pneumonia_indices))]
        else:
            self.labels = pneumonia_mask.values.reshape(-1, 1)
            images = df_metadata['Image Index']

        images = images.apply(lambda x: image_dir / x).values.reshape(-1, 1)
        self.df = pd.DataFrame(np.concatenate((images, self.labels), axis=1), columns=['image', 'label'])
        
        del images

        print("Dataset: {}".format(self.df))
        print("  Number of positive cases: {}".format(sum(self.labels)))
        print("  Number of negative cases: {}".format(len(self.labels) - sum(self.labels)))
            

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
