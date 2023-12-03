import json
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.config import setup_cfg
from src.model import build_model_resnet, build_model_rt, build_model_traj_resnet
from src.dataset import build_dataset_kitti, build_dataset_nuscenes


class Trainer():
    def __init__(self):
        self._init_cfg()
        self._init_log_dir()
        self._init_dataloader()
        self._init_model()
        self._init_optimizer()
        self._init_loss_func()

    def _init_cfg(self):
        # get default cfg
        self.cfg = setup_cfg()
        self.cfg.TRAINING = True

    def _init_log_dir(self):
        if self.cfg.EXP_NAME != "":
            self.cfg.LOG_DIR += "_" + self.cfg.EXP_NAME
        ckpt_dir = os.path.join(self.cfg.LOG_DIR, "ckpt")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        out_dir = os.path.join(self.cfg.LOG_DIR, "out")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        res_dir = os.path.join(self.cfg.LOG_DIR, "res")
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

    def _init_dataloader(self):
        if self.cfg.DATASET == "Kitti":
            self.train_dataloader, self.val_dataloader = build_dataset_kitti(self.cfg, is_train=True)
        elif self.cfg.DATASET == "Nuscenes":
            self.train_dataloader, self.val_dataloader = build_dataset_nuscenes(self.cfg, is_train=True)

    def _init_model(self):
        ## FIXME: parallel the model or not
        if self.cfg.PARALLEL:
            self.device = self.cfg.DEVICE[0]
            self.device_ids = self.cfg.DEVICE
        else:
            self.device = self.cfg.DEVICE

        if self.cfg.MODEL == "Resnet":
            self.model = build_model_resnet(self.cfg,
                                            device=self.device,
                                            num_layers=self.cfg.NUM_LAYERS,
                                            pretrained=self.cfg.PRETRAINED)
        elif self.cfg.MODEL == "RT":
            self.model = build_model_rt(self.cfg,
                                        device=self.device)
        elif self.cfg.MODEL == "TR":
            self.model = build_model_traj_resnet(self.cfg,
                                        device=self.device)

        if self.cfg.PARALLEL:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        if self.cfg.RESUME:
            self.load_checkpoint()

    def _init_optimizer(self):
        if self.cfg.OPTIMIZER == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.LEARNING_RATE)

    def _init_loss_func(self):
        if self.cfg.LOSS_FUNC == "Focal":
            self.loss_func = FocalLoss(alpha=self.cfg.ALPHA, gamma=self.cfg.GAMMA, epsilon=self.cfg.EPSILON)

    def do_train(self, sup=True):
        if sup:
            self.sup_train()
        else:
            self.nesy_train()

    def sup_train(self):
        # prepare
        start_epoch = self.cfg.START_EPOCH
        max_epoch = self.cfg.MAX_EPOCH

        best_train_loss = torch.inf
        best_val_loss = torch.inf

        self.train_loss_curve = []
        self.val_loss_curve = []

        # run
        for epoch in range(start_epoch, max_epoch):
            # train
            train_loss = torch.Tensor([0.]).to(self.device)
            self.model.train()
            for batch in tqdm(self.train_dataloader):
                # load
                images, img_infos = batch['images'].to(self.device), batch["img_infos"]
                grids, trajs = batch['grids'], batch['trajs']

                # forward
                self.optimizer.zero_grad()
                output = self.model(images)

                # backward
                target = torch.stack([torch.Tensor(grid['mat']).float().to(self.device) for grid in grids])
                loss = self.loss_func(output, target)
                train_loss += loss
                loss.backward()

                # update
                self.optimizer.step()

            # log train
            train_loss /= len(self.train_dataloader)
            self.train_loss_curve.append([epoch, train_loss.item()])
            best_train_loss = min(train_loss.item(), best_train_loss)
            print("Training => epoch {}: train loss: {} | best train loss: {}".format(epoch, train_loss.item(), best_train_loss))

            # save checkpoint
            if epoch % self.cfg.CKPT_PERIOD == 0 or epoch == max_epoch-1:
                self.save_checkpoint(epoch=epoch)

            # eval
            if epoch % self.cfg.EVAL_PERIOD == 0 or epoch == max_epoch-1:
                val_loss = torch.Tensor([0.]).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    for batch in tqdm(self.val_dataloader):
                        # load
                        images, img_infos = batch['images'].to(self.device), batch["img_infos"]
                        grids, trajs = batch['grids'], batch['trajs']
                        # forward
                        output = self.model(images)
                        target = torch.stack([torch.Tensor(grid['mat']).float().to(self.device) for grid in grids])
                        loss = self.loss_func(output, target)
                        val_loss += loss
                # log eval
                val_loss /= len(self.val_dataloader)
                self.val_loss_curve.append([epoch, val_loss.item()])
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    self.save_checkpoint(best_val=True)
                print("Evaluation => epoch {}: val loss: {} | best val loss: {}".format(epoch, val_loss.item(),
                                                                                          best_val_loss))

            # save final checkpoint
            if epoch == max_epoch-1:
                self.save_checkpoint(final=True)

        # show
        self. save_loss_curve()

    def nesy_train(self):
        pass

    def save_loss_curve(self, draw=True):
        # save raw data
        out_dir = os.path.join(self.cfg.LOG_DIR, "out")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        loss_curve_path = os.path.join(out_dir, "loss_curve.json")
        loss_curve = {
            "train_loss_curve": self.train_loss_curve,
            "val_loss_curve": self.val_loss_curve
        }
        with open(loss_curve_path, 'w') as f:
            json.dump(loss_curve, f)

        train_loss_curve = np.array(self.train_loss_curve)
        val_loss_curve = np.array(self.val_loss_curve)

        if not draw:
            return

        # draw curve plot and save
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        ax.plot(train_loss_curve[:, 0], train_loss_curve[:, 1], c='orange', marker='o', markersize=1.5, alpha=0.8,
                label="train loss")
        ax.plot(val_loss_curve[:, 0], val_loss_curve[:, 1], c='steelblue', marker='x', markersize=1.5, alpha=0.8,
                label="val loss")
        ax.set_title("Loss Curve")
        ax.grid("False")

        y_ub = max(np.max(train_loss_curve[:, 1]), np.max(val_loss_curve[:, 1]))
        x_ub = np.max(train_loss_curve[:, 0])+1
        plt.ylim(0, y_ub*1.2)
        plt.xlim(0, x_ub)
        plt.legend(loc='best')

        plot_save_path = os.path.join(out_dir, "loss_curve.png")
        plt.savefig(plot_save_path, dpi=100, bbox_inches='tight')
        plt.close()

    def save_checkpoint(self, epoch=0, final=False, best_val=False):
        if best_val:
            ckpt_name = "model_best_valloss.pth"
        elif final:
            ckpt_name = "model_final.pth"
        else:
            ckpt_name = "model_" + ("0000000" + str(epoch))[-7:] + ".pth"

        ckpt_dir = os.path.join(self.cfg.LOG_DIR, "ckpt")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        save_path = os.path.join(ckpt_dir, ckpt_name)
        ## FIXME: save ckpt when paralleling
        torch.save(self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(), save_path)

    def load_checkpoint(self):
        ckpt_path = self.cfg.CKPT
        if os.path.exists(ckpt_path):
            print("Resume from checkpoint: {} ...".format(self.cfg.CKPT))
            ckpt_dict = torch.load(ckpt_path)
            ## FIXME: load ckpt when paralleling
            if hasattr(self.model, "module"):
                self.model.module.load_state_dict(ckpt_dict)
            else:
                self.model.load_state_dict(ckpt_dict)
        else:
            print("The checkpoint path is not valid")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, epsilon=0.0001):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        # negative_weights = torch.pow(1 - target, self.beta)
        loss = 0.

        # clamp the prediction to avoid log(0)
        prediction = torch.clamp(prediction, min=self.epsilon,max=1-self.epsilon)
        #
        positive_loss = -self.alpha * torch.pow(1-prediction, self.gamma) * torch.log(prediction) * positive_index
        negative_loss = -(1-self.alpha) * torch.pow(prediction, self.gamma) * torch.log(1-prediction) * negative_index

        # positive_loss = torch.log(prediction) \
        #                 * torch.pow(1 - prediction, self.alpha) * positive_index
        # negative_loss = torch.log(1 - prediction) \
        #                 * torch.pow(prediction, self.alpha) * negative_weights * negative_index
        #

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        # loss = positive_loss + negative_loss

        if num_positive == 0.:
            loss += negative_loss
        else:
            loss += (positive_loss + negative_loss) / num_positive

        return loss

