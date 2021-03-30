import torch


def train_val_loop(model, epochs:int, train_loader, val_loader, optimizer, criterion, loss_train:list, task:int, bs:int, device, scheduler=None):
    """
        This function will perform train loop (forward-backward pass) and also evaluate performance 
        on validation data after each epoch of training. Finally losses will be printed out.
    Return:
        It returns two list containing training and validation loss
    Args:
        model: neural network model to be train
        epochs: number of epochs(times) train the model over complete train data set
        train_loader: data loader for train set
        val_loader: data loader for validation set
        optimizer: optimizer to update model parameters
        criterion: loss function to evaluate the training through loss
        task: define for which part loop is performed and save the model and results in path for that task
        bs: batch size (number of images grouped in a batch)
        device: device to which tensors will be allocated (in our case, from gpu 0 to 7)
        scheduler: update the learning rate based on chosen scheme if provided
    """
    # store the losses after every epoch 
    loss_train = []
    loss_val = []
    
    for epoch in range(epochs):
        #Training
        model.train()
        running_loss = 0

        for i, samples in enumerate(train_loader):
            inputs = samples['image'].to(device)
            labels = samples['label'].to(device).long()
            # labels = labels.squeeze(1
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            ###accumulating loss for each batch
            running_loss += loss.item()
            
            if scheduler:
                # changing LR
                scheduler.step()

            if i%60 == 0: # intermediate progress printing
                print("epoch{}, iter{}, running loss: {}".format(epoch, i, running_loss/(bs*(i+1))))

        loss_train.append(running_loss/len(train_loader))

        print("epoch{}, Training loss: {}".format(epoch, running_loss/len(train_loader)))
        torch.save(model.state_dict(), f'../weights/T{task}/epoch_{epoch}.pth')

        #Validation
        model.eval()
        running_loss_val = 0
        for i, samples in enumerate(val_loader):
            inputs = samples['image'].to(device)
            labels = samples['label'].to(device).long()
            # labels = labels.squeeze(1)

            with torch.no_grad(): 
                outputs = model(inputs)
                # loss = criterion(outputs,labels.long())
                loss = criterion(outputs,labels)

                ###accumulating loss for each batch
                running_loss_val += loss.item()


            #if i%10 == 0:
        loss_val.append(running_loss_val/len(val_loader))
        print("epoch{}, Validation loss: {}".format(epoch, running_loss_val/len(val_loader)))
        
    return loss_train, loss_val


