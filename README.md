# Semi-I2I
Source codes of "A Semi-Supervised Image-to-Image Translation Framework for SAR–Optical Image Matching" IEEE GRSL

## Prerequisites
- Python 3
- Anaconda 3
- NVIDIA GPU + CUDA cuDNN

## Getting Started
### Training
```
python train.py --model semiD2 --dataroot "root of paired data" --dataroot_u "root of unpaired data" --name "your model's name" --no_dropout --lambda_identity 0 --input_nc 1 --output_nc 1 --lambda_L1 50 --no_html
```

### Test
```
python test.py --dataroot "root of test data" --name "your model's name" --num_test 300 --results_dir "root of your results" --model cycle_gan --input_nc 1 --output_nc 1 --no_dropout --epoch 200
```

## Acknowledgments
This code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
