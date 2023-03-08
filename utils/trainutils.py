import torch

from sklearn.metrics import f1_score

from tqdm.auto import tqdm


def evaluate(model, loader, device):
    
    y_true, y_pred = [], []
    
    with tqdm(total=len(loader)) as pbar:

        for X_batch, y_batch in loader:
            batch_size, seq_len = X_batch.shape[0], X_batch.shape[1]       

            # Move the batch to the correct device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)                

            y_batch_pred = model.inference(X_batch)

            y_true += list(y_batch.cpu())
            y_pred += list(y_batch_pred.cpu())
            
            pbar.update(batch_size)

    return f1_score(y_true, y_pred, average='macro')


def train_epoch(model, loader, optimizer, criterion, device):
    
    # Initialize epoch loss (cummulative loss fo all batchs)
    epoch_loss = 0.0

    with tqdm(total=len(loader)) as pbar:

        for X_batch, y_batch in loader:
            batch_size, seq_len = X_batch.shape[0], X_batch.shape[1]   
            
                
            # Move the batch to the correct device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            log_probs = model(X_batch)                

            # Calculate loss
            loss = criterion(log_probs, y_batch)
            
            ### Pytorch magic! ###
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Keep track of overall epoch loss
            epoch_loss += loss.item()

            pbar.update(batch_size)
            
    return epoch_loss


def train(model, loader_train, loader_test, optimizer, criterion, num_epochs, device, verbose=False):
    
    results = []
    
    print("Total Training Time (total number of epochs: {})".format(num_epochs))
    for epoch in tqdm(range(1, num_epochs+1)):
        model.train()
        epoch_loss = train_epoch(model, loader_train, optimizer, criterion, device)
        model.eval()
        acc_train = evaluate(model, loader_train, device)
        acc_test = evaluate(model, loader_test, device)

        results.append((epoch_loss, acc_train, acc_test))
        
        if verbose is True:
            print("[Epoch {:03d}] loss:\t{:.3f}, f1 train: {:.3f}, f1 test: {:.3f} ".format(epoch, epoch_loss, acc_train, acc_test))
            
    return results