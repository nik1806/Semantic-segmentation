import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_color.png")
    #return filename.endswith("_labelIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, basename+extension)

def image_path_city(root, name):
    return os.path.join(root, name)

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class cityscapes_train(Dataset):

    def __init__(self, root, transform, subset='train' ):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')    
        self.images_root += subset
        self.labels_root += subset
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()
        self.transform = transform
        
        #self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]
        '''if filename[61:-16] != filenameGt[56:-25]:
            print("nooo..")
            print(filename[61:-16])
            print(filenameGt[56:-25])'''
        #filename = '/'.join(filename.split()[2:])
        #with open(image_path_city(self.images_root, filename), 'rb') as f:
        with open(filename, 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            #label = load_image(f).convert('P')
            label = load_image(f).convert('P')        
        oldimage = image

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
            #image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)
    
class cityscapes_val(Dataset):

    def __init__(self, root, transform, subset='val' ):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')    
        self.images_root += subset
        self.labels_root += subset
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()
        self.transform = transform
        
        #self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]
        '''if filename[61:-16] != filenameGt[56:-25]:
            print("nooo..")
            print(filename[61:-16])
            print(filenameGt[56:-25])'''
        #filename = '/'.join(filename.split()[2:])
        #with open(image_path_city(self.images_root, filename), 'rb') as f:
        with open(filename, 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = load_image(f).convert('P')        
        oldimage = image

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
            #image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)
    
    

