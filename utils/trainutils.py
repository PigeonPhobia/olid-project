import torch

from sklearn.metrics import f1_score, classification_report

from tqdm.auto import tqdm
from IPython.display import clear_output

class Dict2Class():
    
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

# def evaluate(model, loader, device):
    
#     y_true, y_pred = [], []
    

#     for X_batch, y_batch in loader:
#         batch_size, seq_len = X_batch.shape[0], X_batch.shape[1]       

#         # Move the batch to the correct device
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)                

#         y_batch_pred = model.inference(X_batch)

#         y_true += list(y_batch.cpu())
#         y_pred += list(y_batch_pred.cpu())
            

#     return f1_score(y_true, y_pred, average='macro')



# def train_epoch(model, loader, optimizer, criterion, device):
    
#     # Initialize epoch loss (cummulative loss fo all batchs)
#     epoch_loss = 0.0

#     for X_batch, y_batch in loader:
#         batch_size, seq_len = X_batch.shape[0], X_batch.shape[1]   
        
            
#         # Move the batch to the correct device
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)

#         log_probs = model(X_batch)                

#         # Calculate loss
#         loss = criterion(log_probs, y_batch)
        
#         ### Pytorch magic! ###
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Keep track of overall epoch loss
#         epoch_loss += loss.item()

            
#     return epoch_loss


# def train(model, loader_train, loader_test, optimizer, criterion, num_epochs, device, verbose=False):
    
#     results = []
    
#     print("Total Training Time (total number of epochs: {})".format(num_epochs))
#     for epoch in tqdm(range(1, num_epochs+1)):
#         model.train()
#         epoch_loss = train_epoch(model, loader_train, optimizer, criterion, device)
#         model.eval()
#         acc_train = evaluate(model, loader_train, device)
#         acc_test = evaluate(model, loader_test, device)

#         results.append((epoch_loss, acc_train, acc_test))
        
#         if verbose is True:
#             print("[Epoch {:03d}] loss:\t{:.3f}, f1 train: {:.3f}, f1 test: {:.3f} ".format(epoch, epoch_loss, acc_train, acc_test))
            
#     return results

def evaluate(model, loader, device):
    
    y_true, y_pred = [], []

    model.eval()
    for X_batch, y_batch in loader:
        batch_size, seq_len = X_batch.shape[0], X_batch.shape[1]      

        # Move the batch to the correct device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        log_probs = model(X_batch)      
                      

        y_batch_pred = torch.argmax(log_probs, dim=1)

        y_true += list(y_batch.cpu())
        y_pred += list(y_batch_pred.cpu())
            

    model.train()
    return f1_score(y_true, y_pred, average='micro'), classification_report(y_true, y_pred)

def predict(model, loader, device):
    
    y_true, y_pred = [], []

    model.eval()
    for X_batch, y_batch in loader:
        batch_size, seq_len = X_batch.shape[0], X_batch.shape[1]     

        # Move the batch to the correct device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        log_probs = model(X_batch)      
                      
        y_batch_pred = torch.argmax(log_probs, dim=1)

        y_true += list(y_batch.cpu())
        y_pred += list(y_batch_pred.cpu())
            

    model.train()
    return f1_score(y_true, y_pred, average='micro'), classification_report(y_true, y_pred)

def train_epoch(model, loader, optimizer, criterion, device):
    
    # Initialize epoch loss (cummulative loss fo all batchs)
    epoch_loss = 0.0

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
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        optimizer.zero_grad()

        # Keep track of overall epoch loss
        epoch_loss += loss.item()

    return epoch_loss

def train(model, loader_train, loader_test, optimizer, criterion, num_epochs, device, verbose=False):
    
    #scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = num_epochs*len(loader_train))
    
    results = []
    
    best_f1 = 0
    best_epoch = 0
    best_model_dict = None

    epoch_iter = tqdm(range(1, num_epochs+1)) if verbose else range(1, num_epochs+1)
    for epoch in epoch_iter:
        clear_output(wait=True)
        model.train()
        epoch_loss = train_epoch(model, loader_train, optimizer, criterion, device)
        model.eval()
        f1_train, report_train = evaluate(model, loader_train, device)
        f1_test, report_test = evaluate(model, loader_test, device)

        results.append((epoch_loss, f1_train, f1_test))
        
        if f1_test>best_f1:
            best_f1 = f1_test
            best_epoch = epoch
            best_model_dict = model.state_dict()
            best_report = report_test

        
        if verbose:
            print("[Epoch {}] loss: {:.3f},\t F1 train: {:.3f},\t F1 test: {:.3f},\t Best epoch: {:02d}".format(epoch, epoch_loss, f1_train, f1_test, best_epoch))
            print(best_report)
            #plot_training_results(results, num_epochs)

        if epoch-best_epoch > 10:
            if verbose:
                print("No further improvement. Early stop.")
            break
        
    return results, best_model_dict, best_f1, best_report