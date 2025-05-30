Summary: This tutorial provides implementation guidance for preparing the Watercolor dataset for object detection tasks using AutoMM. It covers two installation methods (Python CLI and Bash script), dataset structure, and format requirements. The tutorial helps with dataset setup tasks, including downloading and organizing 2,000 watercolor images (1,000 each for training/testing) in the correct directory structure. Key features include cross-platform installation options, COCO format conversion recommendations, and system requirements specification (7.5GB disk space, SSD preferred). It's particularly useful for developers setting up object detection pipelines with AutoMM MultiModalPredictor and those needing to handle VOC to COCO format conversions.

# AutoMM Detection - Prepare Watercolor Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/object_detection/data_preparation/prepare_watercolor.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/multimodal/object_detection/data_preparation/prepare_watercolor.ipynb)



[Watercolor](https://naoto0804.github.io/cross_domain_detection/) is a small object detection dataset with 1,000 training images and 1,000 testing images,
and has a specific domain, i.e. watercolor images. This dataset will be used to show how to \[Fast Finetune on Custom Dataset].

You need 7.5 GB disk space to download and extract this dataset. SSD is preferred over HDD because of its better performance.
The total time to prepare the dataset depends on your Internet speed and disk performance. For example, it often takes 8 min on AWS EC2 with EBS.

You can download the dataset from its [official project page](https://naoto0804.github.io/cross_domain_detection/).
We also prepared a bash script for one-step downloading the dataset: 
[download_watercolor.sh](https://raw.githubusercontent.com/autogluon/autogluon/master/examples/automm/object_detection/download_watercolor.sh).
Or you can also use our cli tool `prepare_detection_dataset` that can download all datasets mentioned in our tutorials.
This python script is in our code: 
[prepare_detection_dataset.py](https://raw.githubusercontent.com/autogluon/autogluon/master/multimodal/src/autogluon/multimodal/cli/prepare_detection_dataset.py),
and you can also run it as a cli: `python3 -m autogluon.multimodal.cli.prepare_detection_dataset`.

## Download with Python Script

The python script does not show progress bar, but is promised to work on all major platforms.
If you are working on a Unix system and needs a progress bar, try the bash script!

You could either extract it in coco17 folder under current directory by running:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name watercolor
```


or extract it in coco17 folder under a provided output path:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset --dataset_name watercolor --output_path ~/data
```


or make it shorter:

```
python3 -m autogluon.multimodal.cli.prepare_detection_dataset -d watercolor -o ~/data
```


## Download with Bash Script
You could either extract it under current directory by running:

```
bash download_watercolor.sh
```


or extract it under a provided output path:

```
bash download_watercolor.sh ~/data
```


The command line output will show the progress bar:

```
% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                               Dload  Upload   Total   Spent    Left  Speed
4 3713M    4  170M    0     0  9627k      0  0:06:34  0:00:18  0:06:16 11.2M
```


And after it finished, VOC datasets are extracted in folder `watercolor`, it contains

```
Annotations  ImageSets  JPEGImages
```


## Dataset Format

Watercolor is in VOC format. **In Autogluon MultiModalPredictor, we strongly recommend using COCO as your data format instead.
Check [AutoMM Detection - Prepare COCO2017 Dataset](prepare_coco17.ipynb) and [Convert Data to COCO Format](convert_data_to_coco_format.ipynb) for more information
about COCO dataset and how to convert a VOC dataset to COCO.**

However, for fast proof testing we also have limit support for VOC format.
While using VOC format dataset, the input is the root path of the dataset, and contains at least:

```
Annotations  ImageSets  JPEGImages
```


## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../../advanced_topics/customization.ipynb).

## Citation
```
@inproceedings{inoue_2018_cvpr,
    author = {Inoue, Naoto and Furuta, Ryosuke and Yamasaki, Toshihiko and Aizawa, Kiyoharu},
    title = {Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2018}
}
```
