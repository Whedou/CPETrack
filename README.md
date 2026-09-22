# CPETrack

Code for an RGB-T tracker built around cross-level pixel expert learning.

The main components are:

- **Pixel Expert Bank (PEB):** constructs complementary RGB-T pixel experts and aligned expert tokens.
- **Input Pixel Fusion (IPF):** combines pixel experts before the single-stream backbone.
- **Template- and Text-guided Token-wise Expert Injection (TTEI):** routes expert tokens into intermediate search features using template cues and text cues.

The implementation retains parts of the [STTrack](https://arxiv.org/abs/2412.15691) training and testing framework. STTrack's published results are not CPETrack results.

## Setup

Create a Python environment and install dependencies via `install_cpetrack.sh` according to your CUDA version. Modify the local dataset, pretrained weight and text cache paths in configuration files before running the code.

## Training

The experiment configurations are in `experiments/cpetrack/`. For the default RGB-T experiment:

```bash
python tracking/train.py --script cpetrack --config deep_rgbt_256 --save_dir ./output --mode single --nproc_per_node 1
```


## Testing

For LasHeR, adjust the dataset root in `RGBT_workspace/test_rgbt_mgpus.py` and run:

```bash
python RGBT_workspace/test_rgbt_mgpus.py --script_name cpetrack --yaml_name deep_rgbt_256 --dataset_name LasHeR --epoch 45 --threads 4 --num_gpus 1
```

## Acknowledgment

This repository builds on the STTrack tracking framework. Please cite the original work when using that framework:

```bibtex
@inproceedings{sttrack,
  title={Exploiting multimodal spatial-temporal patterns for video object tracking},
  author={Hu, Xiantao and Tai, Ying and Zhao, Xu and Zhao, Chen and Zhang, Zhenyu and Li, Jun and Zhong, Bineng and Yang, Jian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={4},
  pages={3581--3589},
  year={2025}
}
```
