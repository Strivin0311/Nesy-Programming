from src.tester import BaseSmokeTester, default_arg_parse, BaseBinClsTester
from src.dataset import build_dataset_kitti


class KittiSmokeTester(BaseSmokeTester):
    def __init__(self, args):
        super(KittiSmokeTester, self).__init__(args)

    def _custom_init_args(self, args):
        # config file
        args.config_file = "./data/kitti-smoke/configs/keypoints_only.yaml"

        # checkpoint to test
        args.ckpt = './data/kitti-smoke/logs/model_best_valloss.pth'

        # image root dir
        args.image_root = './data/kitti-smoke/datasets/kitti/training/image_2'

        # fake bounding box radius for each keypoints
        args.keypoint_radius = 2.0

        # plot settings
        args.plot_show = True
        args.save_plot = True
        args.save_root = "./data/kitti-smoke/figures"

        # result root dir
        args.result_root = "./data/kitti-smoke/results"

    def _custom_init_cfg(self, cfg):
        # for device
        cfg.MODEL.DEVICE = "cuda:1"  # the (single) GPU index we use
        # for testing
        cfg.TEST.DETECTIONS_THRESHOLD = 0.10  # the keypoint confidence threshold
        cfg.TEST.DETECTIONS_PER_IMG = 50  # the max objects to be detected
        cfg.TEST.IMS_PER_BATCH = 1  # the batch size for the test dataloader (had better keep it 1)


class KittiBinClsTester(BaseBinClsTester):
    def __init__(self, args):
        super(KittiBinClsTester, self).__init__(args)

    def _custom_init_args(self, args):
        super()._custom_init_args(args)

        # config file
        args.config_file = "./data/kitti-smoke/configs/keypoints_only.yaml"

        # checkpoint to test
        # args.ckpt = './data/kitti-smoke/logs/model_best_valloss.pth'
        args.ckpt = ""

        # image root dir
        args.image_root = './data/kitti-smoke/datasets/kitti/training/image_2'
        args.save_root = "./data/kitti-smoke/figures"

        # result root dir
        args.result_root = "./data/kitti-smoke/results"

    def _custom_init_cfg(self, cfg):
        super()._custom_init_cfg(cfg)

        # for device
        cfg.MODEL.DEVICE = "cuda:0"  # the (single) GPU index we use
        # for testing
        cfg.TEST.DETECTIONS_THRESHOLD = 0.62  # the keypoint confidence threshold

    def _init_dataloader(self):
        ## build the test dataloader for kitti dataset
        kitti_testloader = build_dataset_kitti(self.cfg, is_train=False)

        return kitti_testloader


if __name__ == "__main__":
    # parse the default arguments
    args = default_arg_parse()

    # build the tester
    # tester = KittiSmokeTester(args)
    tester = KittiBinClsTester(args)

    # do testing (keep test_num_batch None unless you only want to check if the testing process can run through)
    tester.do_test(test_num_batch=5)
