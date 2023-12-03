import os
import sys
sys.path.append('..')

import argparse
import torch
import time

from dataset import DisjDataset
from tok_emb import ParityEmb
from network import testNN
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from mingpt.utils import set_seed


def main(args):
    # generate the name of this experiment

    #############
    # Seed everything for reproductivity
    #############
    set_seed(args.seed)

    #############
    # Load data
    #############

    train_dataset, test_dataset = DisjDataset(seq_len=args.seq_len, is_train=True, n_train=args.n_train), \
    DisjDataset(seq_len=args.seq_len, is_train=False, n_train=args.n_train)

    train_dataset_ulb = None
    print(f'[parity-{args.seq_len}] use {len(train_dataset)} for training and {len(test_dataset)} for testing')

    #############
    # Construct a GPT model and a trainer
    #############
    # vocab_size is the number of different digits in the input, not used if tok_emb is specified
    mconf = GPTConfig(
        vocab_size=1, block_size=args.seq_len, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        num_classes=2, causal_mask=False, losses=args.loss, n_recur=args.n_recur, all_layers=args.all_layers,
        tok_emb=ParityEmb, hyper=args.hyper)
    model = GPT(mconf)

    tconf = TrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        label_size=args.batch_size,
        learning_rate=args.lr,
        lr_decay=args.lr_decay,
        warmup_tokens=1024, # until which point we increase lr from 0 to lr; lr decays after this point
        final_tokens=100 * len(train_dataset), # at what point we reach 10% of lr
        eval_funcs=[testNN], # test without inference trick
        eval_interval=args.eval_interval, # test for every eval_interval number of epochs
        gpu=args.gpu,
        heatmap=args.heatmap,
        prefix=None,
        wandb=None,
        ckpt_path=os.path.join("./logs",
                               f"model_s{args.seq_len}_l{args.n_layer}r{args.n_recur}h{args.n_head}_e{args.epochs}b{args.batch_size}_t" +
                               time.strftime("%y-%m-%d--%H-%M-%S", time.localtime()) +
                               ".pth"
                               )

    )

    trainer = Trainer(model, train_dataset, train_dataset_ulb, test_dataset, tconf)

    #############
    # Start training
    #############
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--eval_interval', type=int, default=1, help='Compute accuracy for how many number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--label_size', type=int, default=16, help='The number of labeled training data in a batch')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--lr_decay', default=False, action='store_true', help='use lr_decay defined in minGPT')
    # Model and loss
    parser.add_argument('--n_layer', type=int, default=1, help='Number of sequential self-attention blocks.')
    parser.add_argument('--n_recur', type=int, default=16, help='Number of recurrency of all self-attention blocks.')
    parser.add_argument('--n_head', type=int, default=4, help='Number of heads in each self-attention block.')
    parser.add_argument('--n_embd', type=int, default=128, help='Vector embedding size.')
    parser.add_argument('--loss', default=[], nargs='+', help='specify constraint losses in \{\}')
    parser.add_argument('--all_layers', default=True, action='store_true', help='apply losses to all self-attention layers')
    parser.add_argument('--hyper', default=[1, 0.1], nargs='+', type=float, help='Hyper parameters: Weights of [L_sudoku, L_attention]')

    # Data
    parser.add_argument('--seq_len', type=int, default=20, help='length of bit string', choices=[20, 40, 60, 80, 100, 200])
    parser.add_argument('--n_train', type=int, default=9000, help='The number of data for train')
    parser.add_argument('--n_test', type=int, default=1000, help='The number of data for test')
    # Other
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproductivity.')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu index; -1 means using all GPUs or using CPU if no GPU is available')
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
    parser.add_argument('--wandb', default=False, action='store_true', help='save all logs on wandb')
    parser.add_argument('--heatmap', default=False, action='store_true', help='save all heatmaps in trainer.result')
    parser.add_argument('--comment', type=str, default='', help='Comment of the experiment')
    args = parser.parse_args()

    # we do not log onto wandb in debug mode
    if args.debug: args.wandb = False

    #FIXME: lazy args setting
    args.gpu = 0
    args.epochs = 2
    args.batch_size = 64

    args.seq_len = 20
    args.n_embd = 128
    args.lr = 2e-3

    main(args)