import torch

def accuracy(logits, y):
    pred = torch.log_softmax(logits, dim=1)
    # if pred > threshold then positive class
    correct = pred.argmax(dim=1).eq(y).sum().item()  # add theshold
    total = len(y)
    acc = correct / total
    return acc