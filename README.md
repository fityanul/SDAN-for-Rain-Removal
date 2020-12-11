# SDAN-for-Rain-Removal (SSDRNet)
If you find this github useful in your research, please consider citing:

"Sequential Dual Attention Network for Rain Streak Removal in a Single Image"

C. -Y. Lin, Z. Tao, A. -S. Xu, L. -W. Kang and F. Akhyar, "Sequential Dual Attention Network for Rain Streak Removal in a Single Image," in IEEE Transactions on Image Processing, vol. 29, pp. 9250-9265, 2020, doi: 10.1109/TIP.2020.3025402

Paper: https://ieeexplore.ieee.org/document/9206069

In this GitHub pages, we provide our model (Sequential dual attention-based Single image DeRaining deep Network/SSDRNet) demo for rain removal in a single image in conjunction with our research paper. We develop our program with adopted the code from: https://github.com/csdwren/PReNet. We test our model performance used 8 datasets from:

1. Rain100L: https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html
2. Rain100H: https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html
3. RainLight: https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html
4. RainHeavy: https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html
5. Rain12: http://yu-li.github.io/paper/li_cvpr16_rain.zip
6. RainDDN: https://xueyangfu.github.io/projects/cvpr2017.html
7. RainDID: https://github.com/hezhangsprinter/DID-MDN
8. Real-World: https://stevewongv.github.io/derain-project.html

We also provided the direct link download of all test datasets above at:
https://drive.google.com/drive/folders/1zRda-AqxrimzpPvleFHYtYM2rykNNZ-s?usp=sharing

## Requirements
1. Ubuntu 16.04, cuda 10.1, cuDNN v-7.5
2. Python3.6
3. Pytorch 1.6
4. h5py
5. opencv-python
6. scikit-image
7. Pillow
8. tensorboardX

## How to run
Run "bash test.sh" or following comments:
- Rain100L:
python test.py --logdir logs/Rain100L/ --save_path results/Rain100L/ --data_path datasets/test/Rain100L/rainy_images/

- Rain100H:
python test.py --logdir logs/Rain100H/ --save_path results/Rain100H/ --data_path datasets/test/Rain100H/rainy_images/

- Rain12:
python test.py --logdir logs/Rain100L/ --save_path results/Rain12/ --data_path datasets/test/Rain12/rainy_images/

- RainLight:
python test.py --logdir logs/RainLight/ --save_path results/RainLight/ --data_path datasets/test/RainLightTest/rainy_images/

- RainHeavy:
python test.py --logdir logs/RainHeavy/ --save_path results/RainHeavy/ --data_path datasets/test/RainHeavyTest/rainy_images/

- RainDDN:
python test.py --logdir logs/RainTrainDDN/ --save_path results/DDN/ --data_path datasets/test/DDNTest/rainy_images/

- RainDID:
python test.py --logdir logs/RainTrainDID/ --save_path results/DID/ --data_path datasets/test/DIDTest/rainy_images/

- Real_finetune:
python test.py --logdir logs/Real/finetune/ --save_path results/Real/finetune/ --data_path datasets/test/Real/rainy_images/

- Real_no_finetune:
python test.py --logdir logs/Real/no_finetune/ --save_path results/Real/no_finetune/ --data_path datasets/test/Real/rainy_images/

## Testing results
All testing results of our model also available at:
https://drive.google.com/drive/folders/1T7IMKbu6oP3bVPF2ymB2j3t4rTOnZyhO?usp=sharing
