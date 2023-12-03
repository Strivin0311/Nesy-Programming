from src.tester import BaseSmokeTester, default_arg_parse, BaseBinClsTester
from src.dataset import build_dataset_nuscenes


class NuscenesSmokeTester(BaseSmokeTester):
    def __init__(self, args):
        super(NuscenesSmokeTester, self).__init__(args)

    def _custom_init_args(self, args):
        # config file
        args.config_file = "./data/nuscenes-smoke/configs/keypoints_only.yaml"

        # checkpoint to test
        args.ckpt = './data/nuscenes-smoke/logs/model_best_valloss.pth'

        # image root dir
        args.image_root = './data/nuscenes-smoke/trainval_datasets/image'

        # fake bounding box radius for each keypoints
        args.keypoint_radius = 2.0

        # plot settings
        args.plot_show = True
        args.save_plot = True
        args.save_root = "./data/nuscenes-smoke/figures"

        # result root dir
        args.result_root = "./data/nuscenes-smoke/results"

    def _custom_init_cfg(self, cfg):
        # for device
        cfg.MODEL.DEVICE = "cuda:0"  # the (single) GPU index we use
        # for testing
        cfg.TEST.DETECTIONS_THRESHOLD = 0.10  # the keypoint confidence threshold
        cfg.TEST.DETECTIONS_PER_IMG = 50  # the max objects to be detected
        cfg.TEST.IMS_PER_BATCH = 1  # the batch size for the test dataloader (had better keep it 1)


class NuscenesBinClsTester(BaseBinClsTester):
    def __init__(self, args):
        super(NuscenesBinClsTester, self).__init__(args)

    def _custom_init_args(self, args):
        super()._custom_init_args(args)

        # config file
        args.config_file = "./data/nuscenes-smoke/configs/keypoints_only.yaml"

        # checkpoint to test
        args.ckpt = './data/nuscenes-smoke/logs/model_best_valloss.pth'

        # image root dir
        args.image_root = './data/nuscenes-smoke/trainval_datasets/image'
        args.save_root = "./data/nuscenes-smoke/figures"

        # result root dir
        args.result_root = "./data/nuscenes-smoke/results"

    def _custom_init_cfg(self, cfg):
        super()._custom_init_cfg(cfg)

        # for device
        cfg.MODEL.DEVICE = "cuda:0"  # the (single) GPU index we use
        # for testing
        cfg.TEST.DETECTIONS_THRESHOLD = 0.10  # the keypoint confidence threshold

    def _init_dataloader(self):
        ## build the test dataloader for Nuscenes dataset
        nuscenes_testloader = build_dataset_nuscenes(self.cfg, is_train=False)

        return nuscenes_testloader


if __name__ == "__main__":
    # parse the default arguments
    args = default_arg_parse()

    # build the tester
    # tester = NuscenesSmokeTester(args)
    tester = NuscenesBinClsTester(args)

    # do testing (keep test_num_batch None unless you only want to check if the testing process can run through)
    tester.do_test(test_num_batch=5)
