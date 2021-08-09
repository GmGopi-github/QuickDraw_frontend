from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class customDatasetClass(Dataset):

    def __init__(self, path):

        """
        Custom Dataset Iterator class
        :param path: Path to the directory containing the images
        Data
        ├───Class0
        │       image0.png
        │       image1.png
        │       image2.png
        │       image3.png
        │
        ├───Class1
        │       image0.png
        │       image1.png
        │       image2.png
        │       image3.png
        │
        ├───Class2
        │       image0.png
        │       image1.png
        │       image2.png
        │       image3.png
        │
        └───Class3
                image0.png
                image1.png
                image2.png
                image3.png
        """
        self.path = path
        self.allImages = []
        self.allTargets = []
        self.allClasses = sorted(os.listdir(self.path))

        for targetNo, targetI in enumerate(self.allClasses):
            for imageI in sorted(os.listdir(self.path + '/' + targetI+'/image/')):
                self.allImages.append(self.path + '/' + targetI + '/image/' + imageI)
                self.allTargets.append(targetNo)

    def crop_image(self,image):
        y,x =np.where(image!=0)
        y_min,x_min,y_max,x_max=np.min(y),np.min(x),np.max(y),np.max(x)
        image=image[y_min:y_max,x_min:x_max]
        return image
    def aspect_ratio(self,image):
        max_shape=max(image.shape[0],image.shape[1]) 
        new_image=np.zeros([max_shape,max_shape]) 
        y_min=(max_shape-image.shape[0])//2
        y_max=y_min+image.shape[0]
        x_min=(max_shape-image.shape[1])//2
        x_max=x_min+image.shape[1]
        new_image[y_min:y_max,x_min:x_max]=image
        new_image=Image.fromarray(new_image)
        new_image=new_image.resize((64,64))
        new_image=(np.array(new_image)>0.1).astype(np.float32)
        return new_image
    def process(self,path):
        image=Image.fromarray(plt.imread(path)[:,:,3])
        image=image.resize((64,64))
        finalimage=(np.array(image)>0.1).astype(np.float32)
        finalimage=self.crop_image(finalimage)
        finalimage=self.aspect_ratio(finalimage)
        return finalimage   
    

    def __getitem__(self, item):

        image= self.process(self.allImages[item])
        target=self.allTargets[item]
        return image[None,:,:],target
    def __len__(self):

        return len(self.allImages)