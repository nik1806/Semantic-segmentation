def convert_to_label(prediction):
    """
        Convert model output to class label
    Args:
        prediction: a CxHxW nd-array; here C=21 (0-20; no. of classes); 
                    each C corresponds to probabilites for that class, eg. C=0 contain score at each element in matrix HxW 
    Return: 
        label: a HxW matrix with each element alloted a class (0-20) depending upon maximum value across C dimension
    """

    
    