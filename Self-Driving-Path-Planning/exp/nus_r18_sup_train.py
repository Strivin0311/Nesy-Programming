import argparse
from src.trainer import Trainer


class NuscenesTrainer(Trainer):
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
        self.cfg.MODEL = "Resnet"
        self.cfg.NUM_LAYERS = 18
        self.cfg.PRETRAINED = False
        self.cfg.CKPT = self.args.ckpt
        # train
        self.cfg.TRAINING = True
        self.cfg.DEVICE = "cuda:" + str(self.args.device)
        self.cfg.SEED = self.args.seed
        self.cfg.LEARNING_RATE = self.args.lr
        self.cfg.START_EPOCH = 0
        self.cfg.MAX_EPOCH = self.cfg.START_EPOCH + self.args.num_epochs
        self.cfg.CKPT_PERIOD = 1
        self.cfg.EVAL_PERIOD = 1
        self.cfg.RESUME = True if self.args.ckpt != "" else False
        # focal loss hyper-params
        self.cfg.ALPHA = self.args.alpha
        self.cfg.GAMMA = self.args.gamma

    def parse_arg(self):
        parser = argparse.ArgumentParser(description="ADSPP training for Nuscens Dataset in a supervised way")
        parser.add_argument('--device', default=0, type=int, help="cuda device number")
        parser.add_argument('--seed', default=1, type=int, help="random seed")
        parser.add_argument('--batch_size', default=16, type=int, help='the size of mini-batch')
        parser.add_argument('--num_epochs', default=100, type=int, help='the number of training epochs')
        parser.add_argument('--ckpt', default='', type=str, help='the checkpoint path')
        parser.add_argument('--lr', default=2e-4, type=float, help='the step size of learning')
        parser.add_argument('--alpha', default=0.75, type=float, help='alpha in focal loss')
        parser.add_argument('--gamma', default=2.0, type=float, help='gamma in focal loss')
        parser.add_argument('--exp_name', default='nus-r18-sup-train', type=str,
                            help='the experiment name which is used for log dir name')
        args = parser.parse_args()

        args.num_epochs = 3

        self.args = args


if __name__ == "__main__":
    trainer = NuscenesTrainer()
    trainer.do_train(sup=True)
