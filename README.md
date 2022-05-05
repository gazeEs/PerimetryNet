

# <div align="center"> **PerimetryNet: A multi-scale fine grained deep network for 3D eye gaze estimation using visual field analysis (Pytorch)** </div>

## Paper details
___


## MPIIGaze
We provide the code for test MPIIGaze dataset with leave-one-person-out evaluation.

### Prepare datasets
* Download **MPIIFaceGaze dataset** from [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation).
* Apply data preprocessing from [here](http://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/).
* Store the dataset to your path.

### Train
The training code will be released after the paper is published.

### Test
* Download the pre-trained models from [here](https://drive.google.com/drive/folders/1evuHA_BUi_yGFp64u8DhCAiI0QJh7OOW?usp=sharing) and Store it to *checkpoint/*.
* Run:
```
 python test.py  --dataset mpiigaze --snapshot /checkpoint/ --evalpath evaluation/mpii_test   --batch_size 100 --gpu 1 
```
This means the code will perform leave-one-person-out testing automatically and store the results to *evaluation/mpii_test*.

## References
This repo is inspired by [L2CS-Net](https://github.com/Ahmednull/L2CS-Net).