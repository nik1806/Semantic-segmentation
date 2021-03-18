# dependencies
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

def unNormalize(image, mean, std):
    '''
        Undo the normalization using mean and std for image.
        image (ndarray, type=Float)
    '''

    # IMAGE OPERATIONS  
    # from (approx) [-1,1] to [0,1]
    for i in range(3):
        image[i] = (image[i] * std[i] + mean[i])
    image = image*255.0 # [0, 1] to [0, 255]
    
    return image

def decode_segmap(image, nc=21):
  
    label_colors = np.array([(0, 0, 0),  # 0=background
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motobrbike, 15=person
                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, nc):
        # idx = image == l
        r[image == l] = label_colors[l, 0]
        g[image == l] = label_colors[l, 1]
        b[image == l] = label_colors[l, 2]
        
    img = np.stack([r, g, b], axis=2)
    return img


# +

def pair_transform_train(image, target):
    """
    Perform appropriate transformations on image and segmentation map.
    For segmap, only convert to tensor and apply augmentation, no normalization etc.
    Args:
        image: Original image
        target: Segmentation map
    """
    # transformation for performing augmentation
    resize = transforms.Resize(size=(224, 224))
    image = resize(image)
    target = resize(target)

    # random horizontal flipping
    if(random.random() > 0.4):
        # print("hflip")
        image = TF.hflip(image)
        target = TF.hflip(target)

    # random rotation
    if(random.random() > 0.4):
        angle = random.randint(-10, 10) # angle between -5 to 5 degree
        image = TF.rotate(image, angle)
        target = TF.rotate(target, angle)


    # aug_T = transforms.Compose([
    #     transforms.Resize((128, 128)),
        # transforms.RandomHorizontalFlip(p=0.2),
        # transforms.RandomRotation(10),
        #transforms.CenterCrop(100),
    # ])
    # img, target = aug_T(img), aug_T(target)

    img_T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # transformations particular for image, e.g., normalization
    image = img_T(image)
    target = torch.from_numpy(np.array(target)).long() # convert to tensor
    return image, torch.clamp(target, max=34)
    # return image, target


# -

def pair_transform_val(image, target):
    """
    Perform appropriate transformations on image and segmentation map.
    For segmap, only convert to tensor and apply augmentation, no normalization etc.
    Args:
        image: Original image
        target: Segmentation map
    """
    # transformation for performing augmentation
    resize = transforms.Resize(size=(256, 256))
    image = resize(image)
    target = resize(target)

    # transformations particular for image, e.g., normalization
    img_T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = img_T(image)

    target = torch.from_numpy(np.array(target)).long() # convert to tensor
    return image, torch.clamp(target, max=34)
