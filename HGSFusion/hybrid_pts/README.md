# Generate hybrid radar points

This tutorial will guide you on how to generate hybrid radar points on VoD and TJ4DRadSet.

## Installation

1. The environment is the same as [Mask2Former](https://github.com/facebookresearch/Mask2Former). 
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.7 and torchvision that matches the PyTorch installation. Install them together at pytorch.org to make sure of this. Note, please check PyTorch version matches that is required by Detectron2.
- Detectron2: follow Detectron2 installation instructions.
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`
2. Copy `hybrid_radar_pts_vod.py` and `hybrid_radar_pts_tj4d.py` to the `/demo`.
3. Download the weight from [here](https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/instance/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_dfa862.pkl). And place it at `./ckpts/`

## Generate hybrid radar points
You can generate the code with the following command.
```
python ./demo/hybrid_radar_pts_vod.py --pts_save_path /path/to/your/pts
```