def unNormalize(image, mean, std):
    '''
        Undo the normalization using mean and std for image.
        image (ndarray, type=Float)
    '''


    # IMAGE OPERATIONS  
    # image = ['image']
    # from (approx) [-1,1] to [0,1]
    for i in range(3):
        image[i] = (image[i] * std[i] + mean[i])
    image = image*255.0 # [0, 1] to [0, 255]
    
    # # KEYPOINT OPERATIONS  
    # keypts = sample['keypoints']
    # keypts = keypts * 0.5 + 0.5 # from [-1, 1] -> [0, 1] 
    # max_keypts = sample['image'].shape[1] # maximum possible value for keypoints
    # keypts *= max_keypts # changing value range from [0, 1] -> [0, max_keypts] 
    # sample['keypoints'] = keypts

    return image