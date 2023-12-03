import json
import os
import argparse
from typing import Union
from tqdm import tqdm
import torch
import math
import numpy as np
from src.trainer import Trainer


class NuscenesRTTrainer(Trainer):
    def __init__(self):
        self.parse_arg()
        super().__init__()

    def _init_cfg(self):
        super()._init_cfg()
        ## custom the config

        # experiment name
        self.cfg.EXP_NAME = self.args.exp_name
        # output shape, choosing from (10,10), (50,50), (100,100)
        self.cfg.OUTPUT_WIDTH = 10
        self.cfg.OUTPUT_HEIGHT = 10
        # data set
        self.cfg.DATASET = "Nuscenes"
        # data loader
        self.cfg.BATCH_SIZE = self.args.batch_size
        self.cfg.SHUFFLE = True
        # model, default using resnet18 as backbone
        self.cfg.MODEL = "RT"
        self.cfg.CKPT = self.args.ckpt
        # train
        self.cfg.TRAINING = True
        self.cfg.PARALLEL = self.args.parallel
        if self.cfg.PARALLEL:
            if not isinstance(self.args.device, Union[list, tuple]):
                raise TypeError("The device should be a idx list/tuple under parallel mode")
            self.cfg.DEVICE = list(self.args.device)
        else:
            if isinstance(self.args.device, list):
                self.cfg.DEVICE = "cuda:" + str(self.args.device[0])
            else:
                self.cfg.DEVICE = "cuda:" + str(self.args.device)
        self.cfg.SEED = self.args.seed
        self.cfg.LEARNING_RATE = self.args.lr
        self.cfg.START_EPOCH = 0
        self.cfg.MAX_EPOCH = self.cfg.START_EPOCH + self.args.num_epochs
        self.cfg.CKPT_PERIOD = 1
        self.cfg.EVAL_PERIOD = 1
        self.cfg.RESUME = True if self.args.ckpt != "" else False

    def sup_train(self):
        ## prepare
        start_epoch = self.cfg.START_EPOCH
        max_epoch = self.cfg.MAX_EPOCH
        model, config = self.model, self.model.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(self.cfg)
        encode_st = self.args.encode_st

        def get_point_mat(grids):
            # boundary and dimensions
            xmin, ymin, xmax, ymax = int(np.min(grids[0]['bx'])), int(np.min(grids[0]['by'])), \
                int(np.max(grids[0]['bx'])), int(np.max(grids[0]['by']))
            width, height = xmax - xmin, ymax - ymin

            sp = torch.zeros((len(grids), self.cfg.OUTPUT_HEIGHT, self.cfg.OUTPUT_WIDTH)).to(
                self.device)
            tp = torch.zeros((len(grids), self.cfg.OUTPUT_HEIGHT, self.cfg.OUTPUT_WIDTH)).to(
                self.device)

            for i, grid in enumerate(grids):
                sx_idx, sy_idx = max(0, min(round(grid['sx'] - xmin), width - 1)), max(0, min(height - 1 - round(
                    grid['sy'] - ymin), height - 1))
                tx_idx, ty_idx = max(0, min(round(grid['tx'] - xmin), width - 1)), max(0, min(height - 1 - round(
                    grid['ty'] - ymin), height - 1))
                sp[i, sy_idx, sx_idx] = 1.
                tp[i, ty_idx, tx_idx] = 1.

            return sp, tp

        #####   run one epoch   ###
        def run_epoch():
            is_train = True
            model.train(is_train)

            def get_acc(logits, target, part="sol"):
                if part == "sol":
                    dim = 0
                elif part == "per":
                    dim = 1
                else:
                    dim = 0

                logit = logits[:, dim, :, :].squeeze(dim=1)  # shape = (batch_size, 10, 10, 2)
                pred = logit.argmax(dim=-1)  # shape = (batch_size, 10, 10)
                tp = torch.logical_and(pred == target, target > 0).sum().item()
                t, p = (target > 0).sum().item(), (pred > 0).sum().item()

                pre = tp / p if p > 0 else 0  # get precision
                rec = tp / t if t > 0 else 0  # get recall

                return pre, rec

            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            for it, batch in pbar:
                ## load data and generate the input/output
                x, img_infos = batch['images'].to(self.device), batch["img_infos"]
                grids, trajs = batch['grids'], batch['trajs']

                y = torch.zeros((x.shape[0], 2, self.cfg.OUTPUT_HEIGHT, self.cfg.OUTPUT_WIDTH)).long().to(
                    self.device)  # shape = (batch_size, 2, 10, 10)
                # solving part: 0-1 matrix where 1 represents the trajectory point and it's given
                sol_y = torch.stack([torch.Tensor(traj['mat']).long().to(self.device) for traj in
                                     trajs])  # shape = (batch_size, 10, 10)
                y[:, 0, :, :] = sol_y
                # perception part: 0-1 matrix where 1 represents the obstacle point while it's not given for symbol grounding
                per_y = torch.stack([torch.Tensor(grid['mat']).long().to(self.device) for grid in
                                     grids])  # shape = (batch_size, 10, 10)
                y[:, 1, :, :] = -100

                if encode_st:
                    sp, tp = get_point_mat(grids)
                else:
                    sp, tp = None, None

                ## forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss, _ = model(x, y, sp=sp, tp=tp)

                    # get solving acc
                    sol_pre, sol_rec = get_acc(logits, sol_y, part="sol")
                    # get perception acc
                    per_pre, per_rec = get_acc(logits, per_y, part="per")
                    # get loss
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus

                ## backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                ## log on std.out
                pbar.set_description(
                    f"Epoch {epoch + 1} iter {it}: Train loss {loss.item():.5f} | Solving[precision: {sol_pre:.4f}; recall: {sol_rec:.4f}] | Perception[precision: {per_pre:.4f}; recall: {per_rec:.4f}]"
                )

        ######    running   ######
        for epoch in range(start_epoch, max_epoch):
            run_epoch()
            self.save_checkpoint(epoch)

    def parse_arg(self):
        parser = argparse.ArgumentParser(description="ADSPP Training for Kitti Dataset")
        parser.add_argument('--device', default=0, type=Union[int, list], help="cuda device number(s)")
        parser.add_argument('--parallel', default=False, action="store_true", help="random seed")
        parser.add_argument('--seed', default=1, type=int, help="random seed")
        parser.add_argument('--batch_size', default=32, type=int, help='the size of mini-batch')
        parser.add_argument('--num_epochs', default=500, type=int, help='the number of training epochs')
        parser.add_argument('--ckpt', default='', type=str, help='the checkpoint path')
        parser.add_argument('--lr', default=1e-3, type=float, help='the step size of learning')
        parser.add_argument('--encode_st', default=False, action="store_true",
                            help='choose whether to encode the start/target point into the input')
        parser.add_argument('--exp_name', default='nus-rt-sup-train', type=str,
                            help='the experiment name which is used for log dir name')
        args = parser.parse_args()

        # FIXME: lazy args setting
        args.encode_st = True
        args.device = 0
        args.batch_size = 128
        args.num_epochs = 500

        self.args = args


if __name__ == "__main__":
    trainer = NuscenesRTTrainer()
    trainer.do_train(sup=True)
