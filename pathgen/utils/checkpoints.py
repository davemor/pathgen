import torch

def save_checkpoint(epoch, model, optimizer, path):
    print(f"saving checkpoint to {path}")
    state = { 'epoch': epoch, 
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict() }
    torch.save(state, path)

def load_checkpoint(model, optimizer, path):
    print("loading checkpoint")
    state = torch.load(path)
    epoch = state["epoch"]
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    return epoch, model, optimizer, loss