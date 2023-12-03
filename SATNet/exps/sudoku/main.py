#!/usr/bin/env python3
#
# Partly derived from:
#   https://github.com/locuslab/optnet/blob/master/sudoku/train.py 

import argparse
import signal
import sys
import os
import shutil
import csv
import random

import numpy as np
import numpy.random as npr
#import setproctitle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(threshold=1000)


import prep_dataset
import models
from loss import permutation_invariant_loss, to_oh


MODES = [
    'nonvisual',
    'visual',
    'train-proofreader-nonvisual',
    'train-satnet-visual-infogan',
    'satnet-visual-infogan-generate-dataset',
    'train-backbone-lenet-supervised',
    'train-proofreader-lenet',
]


class CSVLogger(object):
    def __init__(self, fname):
        self.f = open(fname, 'w')
        self.logger = csv.writer(self.f)

    def log(self, fields):
        self.logger.writerow(fields)
        self.f.flush()

def print_header(msg):
    print('===>', msg)

def setup_state(cuda):
    # For debugging: fix the random seed
    # npr.seed(1)
    # torch.manual_seed(7)
    # random.seed(7)  

    if cuda: 
        print('Using', torch.cuda.get_device_name(0))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.init()
    
def freeze_model(model):
    print(f'freezing model {model.__class__.__name__}.')
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    print(f'unfreezing model {model.__class__.__name__}.')

    for param in model.parameters():
        param.requires_grad = True

def main(
    data_dir='sudoku', 
    boardSz=3, 
    batchSz=40, 
    testBatchSz=40, 
    aux=300, 
    m=600, 
    nEpoch=100, 
    lr=2e-3, 
    load_model=None, 
    load_infogan=None, 
    no_cuda=False, 
    mode=None, 
    to_train=True,
    leak_labels=False,
    num_injected_input_cell_errors=0,
    solvability='any',
    infogan_labels_dir=None,
    experiment_num=0,
):
    assert mode in MODES
    cuda = not no_cuda and torch.cuda.is_available()
    setup_state(cuda)
    
    save = 'sudoku.{}.boardSz{}-aux{}-m{}-lr{}-bsz{}-exp{}'.format(mode,
            boardSz, aux, m, lr, batchSz, experiment_num)
    save = os.path.join('logs', save)
    if os.path.isdir(save): shutil.rmtree(save)
    os.makedirs(save)

    print_header(f'Run saving to {save}')

    print_header('Loading data')

    perm = None

    if mode in (
            'visual',
            'satnet-visual-infogan-generate-dataset',
            'train-proofreader-lenet',
            'train-satnet-visual-infogan',
            'train-backbone-lenet-supervised',
        ):

        use_dataset = ["satnet", "rrn", "sudoku"][0]

        if use_dataset == "satnet":
            train_set, test_set = prep_dataset.get_sudoku_satnet()
        elif use_dataset == "rrn":
            train_set, test_set = prep_dataset.get_sudoku_rrn()
        else:
            train_set, test_set = prep_dataset.get_sudoku_visual(data_dir, cuda, infogan_label_dir=infogan_labels_dir)

        unperm = None

    elif mode in (
            'nonvisual', 
            'train-proofreader-nonvisual', 
        ):
        train_set, test_set, unperm = prep_dataset.get_sudoku_nonvisual(
            data_dir, cuda, perm, num_injected_input_cell_errors, solvability
        )

    else:
        raise ValueError('Unknown mode')

    print_header('Building model')
    if mode in ('visual', ):
        model = models.MNISTSudokuSolver(boardSz**6, aux, m, leak_labels, softmax=False)

        if load_model:
            model.load_from_pieces_if_present(torch.load(load_model))

    elif mode in ('nonvisual',):
        model = models.SudokuSolver(boardSz**6, aux, m, leak_labels, softmax=False)

        if load_model:
            model.load_from_pieces_if_present(torch.load(load_model))

    elif mode == 'train-proofreader-nonvisual':
        model = models.ProofreadSudokuSolver(boardSz**6, aux, m, leak_labels, softmax=True)

        if load_model:
            model.load_from_pieces_if_present(torch.load(load_model))

        freeze_model(model.sudoku_solver)

    elif mode in ('train-satnet-visual-infogan',):
        model = models.InfoGanSudokuSolver(boardSz**6, aux, m, leak_labels)

        if load_model:
            model.load_from_pieces_if_present(torch.load(load_model))

        freeze_model(model.discriminator)
        freeze_model(model.qnet)


    elif mode in ('satnet-visual-infogan-generate-dataset',):
        model = models.InfoGanSudokuSolver(boardSz**6, aux, m, leak_labels)
        
        if load_model:
            model.load_from_pieces_if_present(torch.load(load_model))

    elif mode in ('train-backbone-lenet-supervised',):
        model = models.MNISTSudokuSolver(boardSz**6, aux, m, leak_labels, softmax=True)

        if load_model:
            model.load_from_pieces_if_present(torch.load(load_model))

        freeze_model(model.sudoku_solver)


    elif mode == 'train-proofreader-lenet':
        model = models.MNISTSudokuSolver(boardSz**6, aux, m, leak_labels, softmax=False, proofread=True)

        if load_model:
            model.load_from_pieces_if_present(torch.load(load_model))

        freeze_model(model.digit_convnet)
        freeze_model(model.sudoku_solver.sudoku_solver)


    else:
        raise ValueError('Unknown mode')

    if cuda: model = model.cuda()

    if mode in ('visual',):
        optimizer = optim.Adam([
            {'params': model.sudoku_solver.parameters(), 'lr': lr},
            {'params': model.digit_convnet.parameters(), 'lr': 1e-5},
            ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    train_logger = CSVLogger(os.path.join(save, 'train.csv'))
    test_logger = CSVLogger(os.path.join(save, 'test.csv'))
    fields = ['epoch', 'loss', 'total_board_error', 'per_cell_error', 'digit_classification_error']
    train_logger.log(fields)
    test_logger.log(fields)

    img_list_train = []
    img_list_test = []

    def render_img_list(img_list, file_name):
        # Animation showing the improvements of the generator.
        fig = plt.figure(figsize=(10,10))
        plt.axis("off")
        ims = [[plt.imshow(i, animated=True)] for i in img_list]
        anim = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000, blit=True)
        anim.save(file_name, dpi=80, writer='imagemagick')

    def signal_handler(sig, frame):
    
        render_img_list(img_list_train, 'perm_train.gif')
        render_img_list(img_list_test, 'perm_test.gif')

        sys.exit(0)
    
    if mode in (
        'train-satnet-visual-infogan', 
    ):
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal_handler)


    # run_extract(boardSz, 0, model, optimizer, test_logger, train_set, testBatchSz, unperm, mode)
    label_set = test(boardSz, 0, model, optimizer, test_logger, test_set, testBatchSz, unperm, mode, img_list_test, leak_labels)

    if mode in ('satnet-visual-infogan-generate-dataset', ):
        file_name = 'test_labels.pt'
        print(f'Saving labels with dim {label_set.shape} to {file_name}.')
        torch.save(label_set, file_name)

        label_set = test(boardSz, 0, model, optimizer, test_logger, train_set, testBatchSz, unperm, mode, img_list_test, leak_labels)

        file_name = 'train_labels.pt'
        print(f'Saving labels with dim {label_set.shape} to {file_name}.')
        torch.save(label_set, file_name)
        return save

    if to_train:
        for epoch in range(1, nEpoch+1):
            train(boardSz, epoch, model, optimizer, train_logger, train_set, batchSz, unperm, mode, img_list_train, leak_labels)
            test(boardSz, epoch, model, optimizer, test_logger, test_set, testBatchSz, unperm, mode, img_list_test, leak_labels)
            torch.save(model.get_pieces(), os.path.join(save, 'it'+str(epoch)+'.pth'))

    if mode in (
        'train-satnet-visual-infogan', 
    ):
        signal.signal(signal.SIGINT, original_sigint_handler)
        
        render_img_list(img_list_train, 'perm_train.gif')
        render_img_list(img_list_test, 'perm_test.gif')

    return save
    

def run(boardSz, epoch, model, optimizer, logger, dataset, batchSz, to_train, unperm, mode, img_list, leak_labels):

    loss_final, board_err_final, err_solvable_final, err_unsolvable_final, solvable_total, unsolvable_total, cell_err_final, visual_err_final, num_inputs_final = 0., 0., 0., 0., 0., 0., 0., 0., 0.

    loader = DataLoader(dataset, batch_size=batchSz)
    tloader = tqdm(enumerate(loader), total=len(loader))

    if mode in ('satnet-visual-infogan-generate-dataset',):
        label_set = []


    avg_perm = torch.zeros(len(tloader), 9, 9)
    for i, content in tloader:
        data, is_input, label = content[:3]
        # FIXME vvvv
        solvable = content[3] if len(content) > 3 and mode not in ('train-backbone-lenet-supervised',) else None

        is_input_mono = is_input.view(-1, 9).float().mean(dim=1).type(torch.bool)

        if to_train: optimizer.zero_grad()

        preds, reconstructed = model(data.contiguous(), is_input.contiguous())
        preds = preds.contiguous()

        if mode in ('satnet-visual-infogan-generate-dataset',):
            label_set.append(reconstructed.cpu().detach())


        if mode in (
            'train-satnet-visual-infogan', 
            'satnet-visual-infogan-generate-dataset'
        ):

            
            preds_loss = preds.view(-1, 9)[~is_input_mono, :]
            label_loss = label.view(-1, 9)[~is_input_mono, :]

            preds = preds*(1 - is_input.float())
            preds = preds + reconstructed*is_input.float()


            loss, perm_oh, perm = permutation_invariant_loss(preds_loss, label_loss)
            plt.imsave('perm.png', perm.detach().cpu().numpy())
            avg_perm[i] = perm.detach().cpu()


            # Permute the preds so that they can be evaluated against the labels properly.
            # preds = torch.matmul(to_oh(preds.view(-1, 9)), perm_oh).contiguous().view(-1, 9**3)
            preds = torch.matmul(to_oh(preds.view(-1, 9)), perm_oh).contiguous().view(-1, 9**3)


        elif mode in ('train-backbone-lenet-supervised', ):

            infogan_labels = content[3]
            loss = F.binary_cross_entropy(reconstructed, infogan_labels)

            perm_oh = None
            if hasattr(model, 'current_perm') and model.current_perm is not None:
                perm_oh = model.current_perm.cuda()

            preds = preds*(1. - is_input.float())
            preds = preds + reconstructed*is_input.float()

            preds = torch.matmul(to_oh(preds.view(-1, 9)), perm_oh).contiguous().view(-1, 9**3)


        else:
            perm_oh = None
            if hasattr(model, 'current_perm') and model.current_perm is not None:
                perm_oh = model.current_perm.cuda()

            preds = preds*(1. - is_input.float())
            preds = preds + reconstructed*is_input.float()

            if perm_oh is not None:
                preds = torch.matmul(preds.view(-1, 9), perm_oh).contiguous().view(-1, 9**3)

            if leak_labels:
                preds_loss = preds
                label_loss = label
            else:
                preds_loss = preds.view(-1, 9)[~is_input_mono, :]
                label_loss = label.view(-1, 9)[~is_input_mono, :]

            loss = F.binary_cross_entropy(preds_loss, label_loss)

            preds = to_oh(preds.view(-1, 9)).view(-1, 729)


        if to_train:
            loss.backward()
            optimizer.step()

        board_err = computeErr(preds.data, boardSz, unperm)
            
        # Evaluate per-cell accuracy
        preds = preds.view(-1, 9)

        num_cells = label.shape[1]/9
        cell_err = torch.any(preds != label.view(-1, 9), dim=1)
        visual_err = cell_err[is_input_mono].sum().float()
        num_inputs = is_input_mono.int().sum()

        cell_err = cell_err.sum().float()

        if solvable is not None:
            err_solvable_final += computeErr(preds.view(-1, 9**3)[solvable], boardSz, unperm) if solvable.sum() > 0 else 0
            err_unsolvable_final += computeErr(preds.view(-1, 9**3)[~solvable], boardSz, unperm) if (~solvable).sum() > 0 else 0
            solvable_total += float(solvable.sum())
            unsolvable_total += float((~solvable).sum())

        tloader.set_description('Epoch {} {} Loss {:.4f} Total Board Error: {:.4f} Per-Cell Error: {:.4f} Visual Error {:.4}'.format(epoch, ('Train' if to_train else 'Test '), loss.item(), board_err/batchSz, cell_err/batchSz/num_cells, visual_err/num_inputs))
        loss_final += loss.item()*batchSz
        board_err_final += board_err
        cell_err_final += cell_err
        visual_err_final += visual_err
        num_inputs_final += num_inputs

    loss_final, board_err_final, cell_err_final = loss_final/len(loader.dataset), board_err_final/len(loader.dataset), cell_err_final/len(loader.dataset)/num_cells
    logger.log((epoch, loss_final, board_err_final, float(cell_err_final.detach()), float(visual_err_final.detach()/num_inputs_final)))

    solvability_string = (
        f'Solvable: {err_solvable_final:.0f}/{solvable_total:.0f} ({100.*err_solvable_final/solvable_total if solvable_total > 0 else 0:.2f}%), Unsolvable: {err_unsolvable_final:.0f}/{unsolvable_total:.0f} ({100.*err_unsolvable_final/unsolvable_total if unsolvable_total > 0 else 0:.2f}%)'
        if err_solvable_final + err_unsolvable_final == board_err_final*len(loader.dataset) else None
    )

    title = '--------| TRAINING' if to_train else 'TESTING'
    print(f'{title} SET RESULTS: Average loss: {loss_final:.4f} Total Board Error: {board_err_final:.4f}, Per-Cell Error: {cell_err_final:.4f}, Visual Error: {visual_err_final/num_inputs_final:.4f}. {solvability_string}')

    #print('memory: {:.2f} MB, cached: {:.2f} MB'.format(torch.cuda.memory_allocated()/2.**20, torch.cuda.memory_cached()/2.**20))
    torch.cuda.empty_cache()

    # Avg perm
    avg_perm = avg_perm.mean(dim=0)
    plt.imsave('avg_perm_train.png' if to_train else 'avg_perm_test.png', avg_perm.numpy())
    img_list.append(avg_perm.numpy())

    if to_train and mode in ('train-satnet-visual-infogan',):
        model.current_perm = to_oh(avg_perm).clone()

    if mode in ('satnet-visual-infogan-generate-dataset',):
        # label_set = torch.cat(label_set)
        return label_set





def train(args, epoch, model, optimizer, logger, dataset, batchSz, unperm, mode, img_list, leak_labels):
    return run(args, epoch, model, optimizer, logger, dataset, batchSz, True, unperm, mode, img_list, leak_labels)

@torch.no_grad()
def test(args, epoch, model, optimizer, logger, dataset, batchSz, unperm, mode, img_list, leak_labels):
    return run(args, epoch, model, optimizer, logger, dataset, batchSz, False, unperm, mode, img_list, leak_labels)

@torch.no_grad()
def computeErr(pred_flat, n, unperm):
    if unperm is not None: pred_flat[:,:] = pred_flat[:,unperm]

    nsq = n ** 2
    pred = pred_flat.view(-1, nsq, nsq, nsq)

    batchSz = pred.size(0)
    s = (nsq-1)*nsq//2 # 0 + 1 + ... + n^2-1
    I = torch.max(pred, 3)[1].squeeze().view(batchSz, nsq, nsq)

    def invalidGroups(x):
        valid = (x.min(1)[0] == 0)
        valid *= (x.max(1)[0] == nsq-1)
        valid *= (x.sum(1) == s)
        return ~valid

    boardCorrect = torch.ones(batchSz).type_as(pred)
    for j in range(nsq):
        # Check the jth row and column.
        boardCorrect[invalidGroups(I[:,j,:])] = 0
        boardCorrect[invalidGroups(I[:,:,j])] = 0

        # Check the jth block.
        row, col = n*(j // n), n*(j % n)
        M = invalidGroups(I[:,row:row+n,col:col+n].contiguous().view(batchSz,-1))
        boardCorrect[M] = 0

        if boardCorrect.sum() == 0:
            return batchSz

    return float(batchSz-boardCorrect.sum())
