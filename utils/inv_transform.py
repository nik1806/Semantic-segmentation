import numpy as np

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
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
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