import torch
import torch.nn as nn

import numpy as np
import time

class Coach:
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, device, epochs):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs

        # store
        self.train_loss, self.train_acc = [], []
        self.test_loss, self.test_acc = [], []

    def _train_epoch(self):
        self.model.train()
        dataloader = self.train_loader
        batch_loss, batch_correct = [], 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            output = self.model(X)
            loss = self.loss_fn(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss.append(loss.item())
            pred = torch.argmax(output, dim=1)
            batch_correct += torch.sum(pred == y)

        epoch_loss = np.mean(batch_loss)
        epoch_acc = batch_correct.item() / len(dataloader.dataset)
        return epoch_loss, epoch_acc
    
    def _test_epoch(self):
        self.model.eval()
        dataloader = self.test_loader
        batch_loss, batch_correct = [], 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.loss_fn(output, y)

                batch_loss.append(loss.item())
                pred = torch.argmax(output, dim=1)
                batch_correct += torch.sum(pred == y)

        epoch_loss = np.mean(batch_loss)
        epoch_acc = batch_correct.item() / len(dataloader.dataset)
        return epoch_loss, epoch_acc
    
    def train_test(self):
        start = time.time()
        for epoch in range(self.epochs):
            train_epoch_loss, train_epoch_acc = self._train_epoch()
            test_epoch_loss, test_epoch_acc = self._test_epoch()

            print("epoch: ", epoch+1, "/", self.epochs)
            print("[train] loss: ", train_epoch_loss, ", acc: ", train_epoch_acc, ", time: ", time.time()-start)
            print("[test] loss: ", test_epoch_loss, ", acc: ", test_epoch_acc)

            self.train_loss.append(train_epoch_loss)
            self.train_acc.append(train_epoch_acc)
            self.test_loss.append(test_epoch_loss)
            self.test_acc.append(test_epoch_acc)

class CoachSSL():
    def __init__(self, model, sup_loader, unsup_loader, test_loader, lossfn_sup, lossfn_unsup, weight, optimizer, device, epochs):
        self.model = model
        self.sup_loader = sup_loader
        self.unsup_loader = unsup_loader
        self.test_loader = test_loader
        self.lossfn_sup = lossfn_sup
        self.lossfn_unsup = lossfn_unsup
        self.weight = weight
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs

        # store
        self.train_loss, self.train_acc = [], []
        self.test_loss, self.test_acc = [], []

    def _train_epoch(self):
        self.model.train()
        dataloader = self.sup_loader
        batch_loss, batch_correct = [], 0
        iter_u = iter(self.unsup_loader)
        for batch, (label_X, y) in enumerate(dataloader):
            try:
                unlabel_X1, unlabel_X2 = next(iter_u)
            except StopIteration:
                iter_u = iter(self.unsup_loader)
                unlabel_X1, unlabel_X2 = next(iter_u)

            X, y = torch.cat([label_X, unlabel_X1, unlabel_X2]).to(self.device), y.to(self.device)

            output = self.model(X)

            output_sup = output[:len(label_X)]
            loss_sup = self.lossfn_sup(output_sup, y)

            output_unsup = output[len(label_X):]
            output_1, output_2 = torch.chunk(output_unsup, 2)
            loss_unsup = self.lossfn_unsup(output_2, output_1)

            loss = loss_sup + self.weight *loss_unsup

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss.append(loss_sup.item())
            pred = torch.argmax(output_sup, dim=1)
            batch_correct += torch.sum(pred == y)

        epoch_loss = np.mean(batch_loss)
        epoch_acc = batch_correct.item() / len(dataloader.dataset)
        return epoch_loss, epoch_acc

    def _test_epoch(self):
        self.model.eval()
        dataloader = self.test_loader
        batch_loss, batch_correct = [], 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.lossfn_sup(output, y)

                batch_loss.append(loss.item())
                pred = torch.argmax(output, dim=1)
                batch_correct += torch.sum(pred == y)

        epoch_loss = np.mean(batch_loss)
        epoch_acc = batch_correct.item() / len(dataloader.dataset)
        return epoch_loss, epoch_acc

    def train_test(self):
        start = time.time()
        for epoch in range(self.epochs):
            train_epoch_loss, train_epoch_acc = self._train_epoch()
            test_epoch_loss, test_epoch_acc = self._test_epoch()

            print("epoch: ", epoch+1, "/", self.epochs)
            print("[train] loss: ", train_epoch_loss, ", acc: ", train_epoch_acc, ", time: ", time.time()-start)
            print("[test] loss: ", test_epoch_loss, ", acc: ", test_epoch_acc)

            self.train_loss.append(train_epoch_loss)
            self.train_acc.append(train_epoch_acc)
            self.test_loss.append(test_epoch_loss)
            self.test_acc.append(test_epoch_acc)