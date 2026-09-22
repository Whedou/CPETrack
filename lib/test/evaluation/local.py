from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.network_path = '/data/wsk/project/cpetrack/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/data/wsk/project/cpetrack'
    settings.result_plot_path = '/data/wsk/project/cpetrack/output/test/result_plots'
    settings.results_path = '/data/wsk/project/cpetrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/wsk/project/cpetrack/output'
    settings.segmentation_path = '/data/wsk/project/cpetrack/output/test/segmentation_results'
    settings.youtubevos_dir = ''
    settings.lasher_path = '/data/pudata/LasHeR/TestingSet/testingset/'
    settings.vtuav_path = '/data/pudata/VTUAV'
    return settings
