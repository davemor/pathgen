from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from pathgen.utils.logger import Logger
from pathgen.utils.metrics import accuracy

def fit(model: nn.Module, device: torch.cuda.device,
        train_loader: DataLoader, valid_loader: DataLoader,
        num_epochs: int, optimizer, criterion, scheduler=None,
        acc_thresh=0.5):

        # initalise stats
        log = Logger()
        start_time_sec = time()

        print(f'Fitting model for {num_epochs} epochs')
        for epoch in range(num_epochs):
            # train and evaluate on the training set
            model.train()
            for batch_idx, (X, y) in enumerate(train_loader):
                # put X and y for the batch on the GPU
                X = X.to(device)
                y = y.to(device)

                # forward pass
                logits = model(X)
                loss = criterion(logits, y)

                # backwards pass
                optimizer.zero_grad()
                loss.backward()

                # gradient descent
                optimizer.step()

                # log the metrics
                acc = accuracy(logits, y)
                log('train_acc', acc)
                log('train_loss', loss.item())

                # print out the metrics for the batch (this is overwritten next batch)
                print('\r', f'train.\t\tepoch: {epoch}\tbatch: {batch_idx + 1}/{len(train_loader)}\tloss: {loss:.3f}\taccuracy: {acc:.3f} ', sep='', end='', flush=True)

            # evaluate on the validation set
            model.eval()
            with torch.no_grad():
                for batch_idx, (X, y) in enumerate(valid_loader):
                    # put X and y for the batch on the GPU is possible
                    X = X.to(device)
                    y = y.to(device)

                    # forward pass
                    logits = model(X)
                    loss = criterion(logits, y)

                    # computer the metric and log them
                    acc = accuracy(logits, y)
                    log('valid_acc', acc)
                    log('valid_loss', loss.item())

                    print('\r', f'validate.\tepoch: {epoch}\tbatch: {batch_idx + 1}/{len(valid_loader)}\t\tloss: {loss:.3f}\taccuracy: {acc:.3f} ', sep='', end='', flush=True)        

            log.end_epoch()
            log.print_summary_of_latest_epoch()

            if scheduler:
                scheduler.step()

        # training complete
        end_time_sec       = time()
        total_time_sec     = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / num_epochs
        print("training complete.")
        print('Time total:     %5.2f sec' % (total_time_sec))
        print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))
        log('total_total_time_sec', total_time_sec)

        # output
        return log.history()

def test(model, device, test_loader):
    log = Logger()
    start_time_sec = time()

    # evaluate on the test set
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            # put X and y for the batch on the GPU is possible
            X = X.to(device)
            y = y.to(device)

            # forward pass
            logits = model(X)

            # computer the metric and log them
            acc = accuracy(logits, y)
            log('test_acc', acc)

            print('\r', f'testing batch: {batch_idx + 1}/{len(test_loader)}\taccuracy: {acc:.3f} ', sep='', end='', flush=True)        

    # we just do one epoch for test as we just want a result of the predictions
    log.end_epoch()
    log.print_summary_of_latest_epoch()

    # work out the time
    end_time_sec       = time()
    total_time_sec     = end_time_sec - start_time_sec
    log('total_total_time_sec', total_time_sec)

    return log.history()
