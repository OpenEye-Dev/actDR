# actDR

## Install
- Install Torch7
- Install CUDA libraries and make sure you can used cudnn
- Install the 3rd Party Scripts ```./install_thirdparty.sh```

### Third Party Stuff
The third party scripts are installed as part of the jeepers common workflow scripts.

Added from monitoring tools from [trainplot](https://github.com/joeyhng/trainplot)
There is also a ```transform.lua``` from facebook that does the data augmentation
As jeepers and the install process improves, this should be less confusing.

## Run it

There is some setup required to prepare your data. Below we assume you have a directory above this repository called "kaggle_data", where images are already converted to png and there is a csv with labels.

```
CUDA_VISIBLE_DEVICES=6,7 th train.lua -lr 0.0005 -plotid 1 -out_path_base output/checkpoints/small_training/small_multigpu_test -val 1 -train_dir ../kaggle_data/train_small_90val_png/ -test_dir ../kaggle_data/train_small_10val_png/ -input_size 128 -batch_size 256 -resample_weight 0.9 -pool_size 12 -multi_gpu 1
```

## License

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
