import torch

def evaluate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            if isinstance(X, list):
                X = (X[0].to(device), X[1].to(device))
                y = y.to(device)
            else:
                X, y = X.to(device), y.to(device)
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"validation Accuracy: {(100*correct):>0.01f}%, Avg loss: {test_loss:>8f} \n")
    return correct