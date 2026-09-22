from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.cpetrack.config import cfg, update_config_from_file


def parameters(yaml_name: str, epoch=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/cpetrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(prj_dir, f"output/checkpoints/train/cpetrack/{yaml_name}/CPETrack_ep{epoch:04d}.pth.tar")
    if not os.path.isfile(params.checkpoint):
        # Existing checkpoints remain loadable after the public-facing rename.
        legacy_checkpoint = os.path.join(
            prj_dir, f"output/checkpoints/train/sttrack/{yaml_name}/STTrack_ep{epoch:04d}.pth.tar"
        )
        if os.path.isfile(legacy_checkpoint):
            params.checkpoint = legacy_checkpoint
    print("params.checkpoint:",params.checkpoint)
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
