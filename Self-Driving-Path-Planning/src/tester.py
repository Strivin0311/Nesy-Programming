import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import setup_cfg
from src.model import build_model_resnet
from src.dataset import build_dataset_kitti, build_dataset_nuscenes
from src.astar import AStarPlanner, GridAStarPlanner

class Tester():
    def __init__(self):
        self._init_cfg()
        self._init_log_dir()
        self._init_dataloader()
        self._init_model()
        self._init_planner()
        self._init_eval()

    def _init_cfg(self):
        # get default cfg
        self.cfg = setup_cfg()
        self.cfg.TRAINING = False
        self.cfg.BATCH_SIZE = 1  # fix at 1

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
            self.test_dataloader = build_dataset_kitti(self.cfg, is_train=False)
        elif self.cfg.DATASET == "Nuscenes":
            self.test_dataloader = build_dataset_nuscenes(self.cfg, is_train=False)

    def _init_model(self):
        if self.cfg.MODEL == "Resnet":
            self.model = build_model_resnet(self.cfg,
                                            device=self.cfg.DEVICE,
                                            num_layers=self.cfg.NUM_LAYERS,
                                            pretrained=self.cfg.PRETRAINED)

        self.load_checkpoint()
        self.device = self.cfg.DEVICE
        self.model.eval()

    def _init_planner(self):
        if self.cfg.PLANNER == "Grid":
            self.planner = GridAStarPlanner()
        elif self.cfg.PLANNER == "Astar":
            self.planner = AStarPlanner()

    def _init_eval(self):
        self.det_eval_funcs = {}
        self.det_eval_res = {}
        self.pla_eval_funcs = {}
        self.pla_eval_res = {}

        if "accuracy" in self.cfg.METRICS:
            self.det_eval_funcs["accuracy"] = accuracy_score
        if "precision" in self.cfg.METRICS:
            self.det_eval_funcs['precision'] = precision_score
        if "recall" in self.cfg.METRICS:
            self.det_eval_funcs['recall'] = recall_score
        if "f1" in self.cfg.METRICS:
            self.det_eval_funcs['f1'] = f1_score
        if "collide" in self.cfg.METRICS:
            self.pla_eval_funcs['collide'] = None

    def _det_eval(self, output, target):
        threshold = self.cfg.THRESHOLD
        with torch.no_grad():
            output = torch.squeeze(output, dim=0)
            output = (output >= threshold).float()
            output = output.view(-1).cpu().numpy()
            target = target.view(-1).cpu().numpy()

        for k, eval_func in self.det_eval_funcs.items():
            self.det_eval_res[k] += eval_func(target, output)

    def _pla_eval(self, pr_traj, gt_traj, gt_grid):
        pass

    def _save_eval(self):
        pass

    def _gen_traj(self, output, gt_grid):
        threshold = self.cfg.THRESHOLD
        xmin, ymin, xmax, ymax = int(np.min(gt_grid['bx'])), int(np.min(gt_grid['by'])), \
            int(np.max(gt_grid['bx'])), int(np.max(gt_grid['by']))
        width, height = xmax - xmin, ymax - ymin

        with torch.no_grad():
            output = torch.squeeze(output, dim=0)
            output = output >= threshold
            output = output.float()

        pr_traj = {}
        if self.cfg.PLANNER == "Grid": # planning on the matrix
            # predicted grid is just the output matrix
            pr_grid = output
            # get the start/target point idx on the matrix from the grid
            sx, sy, tx, ty = gt_grid['sx'], gt_grid['sy'], gt_grid['tx'], gt_grid['ty']
            sx_idx, sy_idx = max(0, min(round(sx - xmin), width - 1)), max(0, min(height - 1 - round(
                sy - ymin), height - 1))
            tx_idx, ty_idx = max(0, min(round(tx - xmin), width - 1)), max(0, min(height - 1 - round(
                ty - ymin), height - 1))
            path = np.array(self.planner.planning(pr_grid, sx_idx, sy_idx, tx_idx, ty_idx))[::-1]
            pathx, pathy = list(map(lambda x: min(x + xmin, xmax-1), path[:, 0])),\
                list(map(lambda y: min((height-1-y) + ymin, ymax-1), path[:, 1]))

            mat = np.zeros((height, width))
            for x, y in path:
                mat[y][x] = 1.
            pr_traj = {
                "pathx": pathx, "pathy": pathy,
                "mat": mat.tolist()
            }
        elif self.cfg.PLANNER == "Astar":  # planning on the grid
            # generate the predicted grid from the matrix
            idxs = output.nonzero()
            pr_grid = {}
            pr_grid["sx"], pr_grid["sy"], pr_grid["tx"], pr_grid["ty"], pr_grid["bx"], pr_grid["by"] \
                = gt_grid["sx"], gt_grid["sy"], gt_grid["tx"], gt_grid["ty"], gt_grid["bx"], gt_grid["by"]
            pr_grid["ox"], pr_grid["oy"] = [], []

            ox = [min(int(xi) + xmin, xmax - 1) for xi in idxs[:, 1]]
            oy = [min((height - 1 - int(yi)) + ymin, ymax - 1) for yi in idxs[:, 0]]

            pr_grid["ox"].append(ox)
            pr_grid["oy"].append(oy)

            pathx, pathy, _, _ = self.planner.planning(
                sx=pr_grid['sx'], sy=pr_grid['sy'], gx=pr_grid['tx'], gy=pr_grid['ty'],
                ox=[x for ox in pr_grid['ox'] for x in ox] + pr_grid['bx'],
                oy=[y for oy in pr_grid['oy'] for y in oy] + pr_grid['by'],
                resolution=1, rr=0.0001, save_process=False
            )
            xidx, yidx = list(map(lambda x: max(0, min(round(x-xmin), width-1)), pathx)), \
                list(map(lambda y: max(0, min(height-1-round(y-ymin), height-1)), pathy))
            mat = np.zeros((height, width))
            for x, y in zip(xidx, yidx):
                mat[y][x] = 1.

            pr_traj = {
                "pathx": pathx, "pathy": pathy,
                "mat": mat.tolist()
            }

        return pr_traj

    def do_test(self, sup=True):
        if sup:
            self.sup_test()
        else:
            self.nesy_test()

    def sup_test(self):

        for batch in tqdm(self.test_dataloader):  # batch_size fix at 1
            # forward the model and get output
            images, img_infos, gt_targets, gt_grids, gt_trajs = batch["images"].to(self.device), batch["img_infos"], \
                batch["targets"], batch["grids"], batch["trajs"]
            output = self.model(images)

            # detection evaluate
            target = torch.Tensor(gt_grids[0]["mat"]).float().to(self.device)
            self._det_eval(output, target)

            # planning evaluate
            pr_traj = self._gen_traj(output, gt_grids[0])
            self._pla_eval(pr_traj, gt_trajs[0], gt_grids[0])
        self._save_eval()

    def nesy_test(self):
        pass

    def load_checkpoint(self):
        ckpt_path = self.cfg.CKPT
        if os.path.exists(ckpt_path):
            ckpt_dict = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt_dict)
        else:
            print("The checkpoint path is not valid")



