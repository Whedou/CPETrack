import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse

prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.cpetrack import CPETrack
import lib.test.parameter.cpetrack as rgbt_params
import multiprocessing
import torch
from lib.train.dataset.depth_utils import get_x_frame
import time




def genConfig(seq_path, set_type):
    if set_type == 'RGBT234':
        ############################################  have to refine #############################################
        RGB_img_list = sorted(
            [seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if
                             os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif set_type == 'GTOT':
        ############################################  have to refine #############################################
        exts = {'.png', '.bmp'}

        RGB_img_list = sorted(
            [seq_path + '/v/' + p for p in os.listdir(seq_path + '/v')
             if os.path.splitext(p)[1].lower() in exts]
        )
        T_img_list = sorted(
            [seq_path + '/i/' + p for p in os.listdir(seq_path + '/i')
            # [seq_path + '/i_resized/' + p for p in os.listdir(seq_path + '/i')
             if os.path.splitext(p)[1].lower() in exts]
        )
        # RGB_img_list = sorted(
        #     [seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] == '.png'])
        # T_img_list = sorted(
        #     [seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] == '.png'])

        RGB_gt = np.loadtxt(seq_path + '/groundTruth_v.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')

        x_min = np.min(RGB_gt[:, [0, 2]], axis=1)[:, None]
        y_min = np.min(RGB_gt[:, [1, 3]], axis=1)[:, None]
        x_max = np.max(RGB_gt[:, [0, 2]], axis=1)[:, None]
        y_max = np.max(RGB_gt[:, [1, 3]], axis=1)[:, None]
        RGB_gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

        x_min = np.min(T_gt[:, [0, 2]], axis=1)[:, None]
        y_min = np.min(T_gt[:, [1, 3]], axis=1)[:, None]
        x_max = np.max(T_gt[:, [0, 2]], axis=1)[:, None]
        y_max = np.max(T_gt[:, [1, 3]], axis=1)[:, None]
        T_gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    elif set_type == 'LasHeR':
        RGB_img_list = sorted(
            [seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if p.endswith(".jpg")])
        T_img_list = sorted(
            [seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif 'VTUAV' in set_type:
        def image_key(path):
            stem = os.path.splitext(os.path.basename(path))[0]
            return (0, int(stem)) if stem.isdigit() else (1, stem)

        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        RGB_img_list = sorted(
            [join(seq_path, 'rgb', p) for p in os.listdir(join(seq_path, 'rgb'))
             if os.path.splitext(p)[1].lower() in image_exts], key=image_key)
        T_img_list = sorted(
            [join(seq_path, 'ir', p) for p in os.listdir(join(seq_path, 'ir'))
             if os.path.splitext(p)[1].lower() in image_exts], key=image_key)

        # The paired tracker is initialized in the RGB reference coordinates.
        RGB_gt = np.atleast_2d(np.loadtxt(join(seq_path, 'rgb.txt')))[:, :4]
        T_gt = np.copy(RGB_gt)

    # 新增测试
    elif set_type == 'RGBT210':
        RGB_img_list = sorted(
            [seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible')
             if os.path.splitext(p)[1].lower() == '.jpg']
        )
        T_img_list = sorted(
            [seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared')
             if os.path.splitext(p)[1].lower() == '.jpg']
        )

        RGB_gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')
        T_gt = np.copy(RGB_gt)




    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='prompt'):
    seq_txt = seq_name.rsplit('/', 1)[-1] if 'VTUAV' in dataset_name else seq_name
    # save_name = '{}_ep{}'.format(yaml_name, epoch)
    save_name = '{}'.format(yaml_name)
    save_path = f'./RGBT_workspace/results/{dataset_name}/{save_name}_{epoch}/' + seq_txt + '.txt'
    save_folder = f'./RGBT_workspace/results/{dataset_name}/{save_name}_{epoch}/'
    os.makedirs(save_folder, exist_ok=True)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    if script_name == 'cpetrack':
        params = rgbt_params.parameters(yaml_name, epoch)
        cpetrack = CPETrack(params)
        tracker = CPETrackRGBT(tracker=cpetrack)

    seq_path = join(seq_home, *seq_name.split('/'))
    print('——————————Process sequence: ' + seq_name + '——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
    if not RGB_img_list:
        raise RuntimeError('No images found for {}'.format(seq_name))
    if len(RGB_img_list) != len(T_img_list):
        raise ValueError(
            'RGB/IR frame-count mismatch for {}: {} vs {}'.format(
                seq_name, len(RGB_img_list), len(T_img_list)
            )
        )
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    for frame_idx, (rgb_path, T_path) in enumerate(zip(RGB_img_list, T_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'))
            tracker.initialize(image, RGB_gt[0].tolist(), seq_name=seq_name)  # xywh
        elif frame_idx > 0:
            # track
            image = get_x_frame(rgb_path, T_path, dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'))
            region, confidence = tracker.track(image)  # xywh
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    if not debug:
        np.savetxt(save_path, result)
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class CPETrackRGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region, seq_name=None):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)

        init_info = {
            'init_bbox': list(gt_bbox_np),
            'sequence_name': seq_name,
        }  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='cpetrack',
                        help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='deep_rgbt_256',
                        help='Name of tracking method.')  # vitb_256_mae_ce_32x4_ep300 vitb_256_mae_ce_32x4_ep60_prompt_i32v21_onlylasher_rgbt
    parser.add_argument('--dataset_name', type=str, default='LasHeR',
                        help='Name of dataset (GTOT,RGBT210,RGBT234,LasHeR,VTUAVST,VTUAVLT).')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=13, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    parser.add_argument('--dataset_root', default='', type=str,
                        help='Override dataset root (for example /data/pudata/VTUAV/test_ST).')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    # path initialization
    seq_list = None
    if dataset_name == 'GTOT':
        seq_home = '/data/pudata/GTOT/'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'RGBT234':
        seq_home = '/data/pudata/RGBT234/'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name == 'LasHeR':
        seq_home = '/data/pudata/LasHeR/TestingSet/testingset/testingset'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    elif dataset_name in ('VTUAVST', 'VTUAVLT'):
        default_root = '/data/pudata/VTUAV/test_ST' if dataset_name == 'VTUAVST' else '/data/pudata/VTUAV/test_LT'
        seq_home = args.dataset_root or default_root
        if not os.path.isdir(seq_home):
            raise FileNotFoundError('VTUAV dataset root not found: {}'.format(seq_home))
        seq_list = []
        for current_root, dirs, _ in os.walk(seq_home):
            if 'rgb' in dirs and 'ir' in dirs and os.path.isfile(join(current_root, 'rgb.txt')):
                seq_list.append(os.path.relpath(current_root, seq_home).replace('\\', '/'))
        seq_list.sort()
        if not seq_list:
            raise RuntimeError('No VTUAV sequences found under {}'.format(seq_home))
        leaf_names = [name.rsplit('/', 1)[-1] for name in seq_list]
        if len(set(leaf_names)) != len(leaf_names):
            raise RuntimeError('VTUAV sequence leaf names are not unique; result files would collide.')
    elif dataset_name == 'RGBT210':
        seq_home = '/data/pudata/RGB_T210/'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
        seq_list.sort()
    else:
        raise ValueError("Error dataset!")

    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [
            (s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name) for s
            in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [
            (s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name) for s
            in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time() - start} seconds!")
