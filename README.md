# G-CASCADE - LiTS

Fork of the official Pytorch implementation of [G-CASCADE: Efficient Cascaded Graph Convolutional Decoding for 2D Medical Image Segmentation](https://openaccess.thecvf.com/content/WACV2024/html/Rahman_G-CASCADE_Efficient_Cascaded_Graph_Convolutional_Decoding_for_2D_Medical_Image_WACV_2024_paper.html) WACV 2024. 
This fork has added support for the LiTS dataset, along with the others provided in the official repository.

## Usage:
### Recommended environment:
```
Python 3.10.13
Pytorch 2.1.2
torchvision 0.16.2
```
Please use ```pip install -r requirements.txt``` to install the dependencies.

### Data preparation:
- **Synapse Multi-organ dataset:**
Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and download the dataset. Then split the 'RawData' folder into 'TrainSet' (18 scans) and 'TestSet' (12 scans) following the [TransUNet's](https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md) lists and put in the './data/synapse/Abdomen/RawData/' folder. Finally, preprocess using ```python ./utils/preprocess_synapse_data.py``` or download the [preprocessed data](https://drive.google.com/file/d/1tGqMx-E4QZpSg2HQbVq5W3KSTHSG0hjK/view?usp=share_link) and save in the './data/synapse/' folder. 
Note: If you use the preprocessed data from [TransUNet](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd), please make necessary changes (i.e., remove the code segment (line# 88-94) to convert groundtruth labels from 14 to 9 classes) in the utils/dataset_synapse.py. 

- **ACDC dataset:**
Download the preprocessed ACDC dataset from [Google Drive of MT-UNet](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view) and move into './data/ACDC/' folder.

- **Polyp datasets:**
Download the training and testing datasets [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing) and move them into './data/polyp/' folder.

- **ISIC2018 dataset:**
Download the training and validation datasets from https://challenge.isic-archive.com/landing/2018/ and merge them together. Afterwards, split the dataset into 80%, 10%, and 10% training, validation, and testing datasets, respectively. Move the splited dataset into './data/ISIC2018/' folder. 

- **LITS dataset:**
Download and obtain the dataset in jpg format. Create the following directory structure in the dataset folder. This directory will be the root_path for the train_list and test_lits scripts.
Directory structure -
```text
root_path
├── liver_0
│   ├── images
│   │   └── 45.jpg
│   │   └── 46.jpg
|   |   └── ---
│   └── masks
│       ├── liver
|       │   └── 46.jpg
|       │   └── 47.jpg
|       |   └── ---
│       └── cancer
|           └── 46.jpg
|           └── 47.jpg
|           └── ---
|
|
├── liver_130
│   ├── images
│   │   └── 45.jpg
│   │   └── 46.jpg
|   |   └── ---
│   └── masks
│       ├── liver
|       │   └── 46.jpg
|       │   └── 47.jpg
|       |   └── ---
│       └── cancer
|           └── 46.jpg
|           └── 47.jpg
|           └── ---
```


### Pretrained model:
You should download the pretrained PVTv2 model from [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing), and then put it in the './pretrained_pth/pvt/' folder for initialization. Similarly, you should download the pretrained MaxViT models from [Google Drive](https://drive.google.com/drive/folders/1k-s75ZosvpRGZEWl9UEpc_mniK3nL2xq?usp=share_link), and then put it in the './pretrained_pth/maxvit/' folder for initialization.

### Training:
```
cd into G-CASCADE
```

For Synapse Multi-organ dataset training, run ```CUDA_VISIBLE_DEVICES=0 python -W ignore train_synapse.py```

For ACDC dataset training, run ```CUDA_VISIBLE_DEVICES=0 python -W ignore train_ACDC.py```

For Polyp datasets training, run ```CUDA_VISIBLE_DEVICES=0 python -W ignore train_polyp.py```

For ISIC2018 dataset training, run ```CUDA_VISIBLE_DEVICES=0 python -W ignore train_ISIC2018.py```

For LiTS dataset training, run ```python G-CASCADE-LiTS/train_lits.py --root_path ./LiTS_root --batch_size 16 --max_epochs 200 --num_classes 2 --seed 32 --is_liver --val_log_interval 100 --log_interval 100```

Note that for LiTS, the number of classes is 2, as the code has been implemented to run on liver seperately and tumor seperately. Open source contributions are welcome to change this.
Regarding the training-testing-validation split, we need not mention the paths to each explicitly. The code assumes 131 folders are shown above (from liver_0 to liver_130), and will automatically split the indexes amongst training, testing and validation based on the random seed provided. In order to ensure consistency and avoid overlap of samples in training split and testing split, PLEASE ENSURE that the random seed passed to train_lits.py and test_lits.py is the same.
The is_liver parameter is used to choose either the liver masks or tumor masks. Ensure that while training and later testing, if it is mentioned in one, it should also be mentioned in the other.

### Testing:
```
cd into G-CASCADE 
```

For Synapse Multi-organ dataset testing, run ```CUDA_VISIBLE_DEVICES=0 python -W ignore test_synapse.py```

For ACDC dataset testing, run ```CUDA_VISIBLE_DEVICES=0 python -W ignore test_ACDC.py```

For Polyp dataset testing, run ```CUDA_VISIBLE_DEVICES=0 python -W ignore test_polyp.py```

For ISIC2018 dataset testing, run ```CUDA_VISIBLE_DEVICES=0 python -W ignore test_ISIC2018.py```

For LiTS dataset training, run ```python ./G-CASCADE-LiTS/test_lits.py --is_liver --root_path ./LiTS_root --seed 32 --num_classes 2 --batch_size 16 --test_log_interval 100```

### Credits:

Fork Implemented By -
[Jasmer Singh Sanjotra](https://github.com/TheAlphaJas)
<p> Indian Institute of Technology Indore</p>
 -------------------------------------------------------------------------------------------------------------------------------
 
 
 Original Authors - 
 
[Md Mostafijur Rahman](https://github.com/mostafij-rahman), [Radu Marculescu](https://radum.ece.utexas.edu/)
<p>The University of Texas at Austin</p>
