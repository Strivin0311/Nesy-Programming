"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from helper import visualize_token2token_scores, visualize_cell_attention

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, train_dataset_ulb, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.train_dataset_ulb = train_dataset_ulb
        self.test_dataset = test_dataset
        self.config = config
        self.test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)
        train_dataset_1000 = torch.utils.data.Subset(
            train_dataset,
            list(range(min(1000, len(train_dataset))))
        )
        # dataloaders for evaluation
        self.train_dataloader = DataLoader(train_dataset_1000, batch_size=config.batch_size)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=config.batch_size)
        # dataloaders for training
        if config.label_size < config.batch_size:
            self.loader_lb = DataLoader(self.train_dataset, shuffle=True, pin_memory=True,
                                batch_size=config.label_size,
                                num_workers=config.num_workers)
            self.loader_ulb = DataLoader(self.train_dataset_ulb, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size - config.label_size,
                                num_workers=config.num_workers)
        else:
            self.loader_lb = DataLoader(self.train_dataset, shuffle=True, pin_memory=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)

        self.eval_funcs = [self._test_acc]
        self.eval_interval = config.eval_interval
        self.result = {}
        self.heatmap = config.heatmap
        self.prefix = config.prefix
        self.wandb = config.wandb
        # # we save the attention for the 1st data in the trainloader for every epoch
        # self.atts = [] # a list of atts of shape (num_layers, num_heads, 81, 81)
        for eval_func in self.eval_funcs:
            self.result[eval_func.__name__] = []

        if config.gpu >= 0 and torch.cuda.is_available():
            print(f'Using GPU {config.gpu}')
            self.device = torch.device('cuda', index=config.gpu)
            self.model.to(self.device)
        else:
            # take over whatever gpus are on the system
            self.device = 'cpu'
            if torch.cuda.is_available():
                print('Using all GPUs')
                self.device = torch.cuda.current_device()
                self.model = torch.nn.DataParallel(self.model).to(self.device)
            else:
                print('Using CPU')

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            losses = []

            # for semi-supervised learning
            if is_train and config.label_size < config.batch_size:
                pbar = tqdm(enumerate(self.loader_lb), total=len(self.loader_lb))
                loader_ulb_iterator = iter(self.loader_ulb)
                for it, data_lb in pbar:
                    data_ulb = next(loader_ulb_iterator)
                    x, y = data_lb # (label_size, 81, 28, 28), (label_size, 81)
                    x_ulb = data_ulb[0] # (batch_size - label_size, 81, 28, 28)
                    # place data on the correct device
                    x = x.to(self.device)
                    y = y.to(self.device)
                    x_ulb = x_ulb.to(self.device)

                    # forward the model
                    with torch.set_grad_enabled(is_train):
                        logits, loss, atts = model(x, y, x_ulb)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            else:
                pbar = tqdm(enumerate(self.loader_lb), total=len(self.loader_lb)) if is_train else enumerate(self.test_dataloader)
                for it, (x, y) in pbar:
                    # place data on the correct device
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # forward the model
                    with torch.set_grad_enabled(is_train):
                        logits, loss, atts = model(x, y)
                        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                    if is_train:

                        # backprop and update the parameters
                        model.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()

                        # decay the learning rate based on our progress
                        if config.lr_decay:
                            self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                            if self.tokens < config.warmup_tokens:
                                # linear warmup
                                lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                            else:
                                # cosine learning rate decay
                                progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            lr = config.learning_rate * lr_mult
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                        else:
                            lr = config.learning_rate

                        # report progress
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            return float(np.mean(losses))

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            train_loss = run_epoch('train')
            test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None:
                best_loss = test_loss
                self.save_checkpoint()

            if (epoch + 1) % self.eval_interval == 0:
                print("Testing...")
                train_result = self._test_acc(model, self.train_dataloader)
                test_result = self._test_acc(model, self.test_dataloader)
                self.print_result(epoch, train_result, test_result)

    def _test_acc(self, model, dataloader):
        model.eval()
        acc_cnt = 0
        tot_cnt = 0
        with torch.no_grad():
            for it, (x, y) in enumerate(dataloader):
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device).view(-1) # shape = (batch_size,)

                output, _, _ = model(x) # shape = (batch_size, 2)
                pred = output.argmax(dim=-1) # shape = (batch_size,)
                acc_cnt += (pred == y).sum().item()
                tot_cnt += x.shape[0]

        return acc_cnt, tot_cnt

    def print_result(self, epoch, train_result, test_result):
        print("Accuracy on sampled training dataset and the whole testing dataset")
        print(('{:<6}'+'{:<15}' * 4).format('Epoch', 'train acc(%)', 'test acc(%)',  'train count', 'test count'))
        row_format = '{:<6}' + '{:<15.2f}' * 2 + '{:<15}' * 2
        train_acc_cnt, train_tot_cnt = train_result
        test_acc_cnt, test_tot_cnt = test_result
        print(row_format.format(epoch, 100 * train_acc_cnt / train_tot_cnt, 100 * test_acc_cnt / test_tot_cnt,
                                f'{train_acc_cnt}/{train_tot_cnt}', f'{test_acc_cnt}/{test_tot_cnt}'))

