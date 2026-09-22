class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/wsk/project/cpetrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/wsk/project/cpetrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/wsk/project/STTrack/pretrained_networks'
        self.lasher_dir = '/data/pudata/LasHeR/TrainingSet/train/trainingset/'
        self.vtuav_train_dir = '/data/pudata/VTUAV/train_ST/train_ST/'
