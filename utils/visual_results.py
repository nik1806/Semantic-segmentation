import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.custom_transforms import unNormalize, decode_segmap


def visualize(model, train_loader, save_path:str, device, cls_num:int=19, disp_num:int=10):
    """
        It display pairs of the original image,the ground truth mask and prediction mask (from model).
        For mask each label is color coded for display for better intutive understanding.
    Args:
        model: neural network model to be train
        train_loader: data loader for train set
        save_path: path to save the plot
        device: device to which tensors will be allocated (in our case, from gpu 0 to 7)
        cls_num: number of classes in dataset, parameter for decode_segmap
        disp_num: number of result pairs to display

 
    """
    # model in evaluation model -> batchnorm, dropout etc. adjusted accordingly
    model.eval()
    # iterator on training data
    data = iter(train_loader)
    # init figure object
    fig = plt.figure(figsize=(10,40))
    pred_rgb = list()
    # for img, label in trainloader:
    for i in range(disp_num):
        sample = next(data) # next batch
        imgs, labels = sample['image'], sample['label']
        # img, label = img.to(device).unsqueeze(0), label.to(device) # to gpu
        # using just one image
        img = imgs[0].to(device).unsqueeze(0) # to gpu & add dummy batch dim
        # gnd = np.asarray(label[0]) 
        # deactivate autograd engine - reduce memory usage 
        with torch.no_grad(): 
            pred = model(img) # forward pass
            pred = pred.squeeze(0) # remove extra dimension of batch
            # extract most probable class through C-dim 
            pred_label = torch.argmax(pred, dim=0).cpu().numpy()
            # convert labels to color code
            pred_rgb = decode_segmap(pred_label, nc=cls_num, dataset='cityscapes')

            # plotting
            # original image
            fig.add_subplot(10, 3, 3*i+1)
            img = imgs[0].data.cpu().numpy() # data in image and current form of matrix
            img = unNormalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # unNormalize
            img = img.transpose((1,2,0)).astype(np.uint8) # change dtype to correct format for display
            plt.title('Original')
            plt.imshow(img) # original
            plt.axis('off')
            # ground truth
            fig.add_subplot(10, 3, 3*i+2)
            label = labels[0].data.numpy() # data in image and current form of matrix
            label = decode_segmap(label, nc=cls_num, dataset='cityscapes')
            plt.title('Ground truth')
            plt.imshow(label) 
            plt.axis('off')
            # prediction
            fig.add_subplot(10, 3, 3*i+3)
            plt.title('Prediction')
            plt.imshow(pred_rgb.astype(np.uint8))
            plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight')    
    plt.plot()



