# UHR-DETR

**UHR-DETR: Efficient End-to-End Small Object Detection for Ultra-High-Resolution Remote Sensing Imagery**


## Installation

```bash
conda create -n uhr-detr python=3.8 -y
conda activate uhr-detr
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
pip install -v -e .
pip install psutil
```

> Note: CUDA 11.8 is used as an example. Adjust the PyTorch `--index-url` according to your CUDA version.

## Quick Start

Train:

```bash
python tools/train.py UHR_DETR/configs/UHR_DETR/UHR_DETR_r50_STAR_sub8k.py
```

Test:

```bash
python tools/test.py <config>.py <checkpoint>.pth --work-dir <output_dir>
```

## Citation

```bibtex
@misc{uhr_detr,
      title={UHR-DETR: Efficient End-to-End Small Object Detection for Ultra-High-Resolution Remote Sensing Imagery}, 
      author={Jingfang Li and Haoran Zhu and Wen Yang and Jinrui Zhang and Fang Xu and Haijian Zhang and Gui-Song Xia},
      year={2026},
      eprint={2604.21435},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.21435}, 
}
```

## Acknowledgements

This project is built upon [MMDetection](https://github.com/open-mmlab/mmdetection) and [RT-DETR for MMDetection](https://github.com/flytocc/rtdetr-mmdet).
