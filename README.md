# bidcap
Benchmark Image Dataset Collection And Preprocessing

## Image datasets incorporated so far

* MNIST dataset and its variants - 12000 train, 50000 test
    * MB: MNIST basic dataset
    * MBI: MNIST background image - A patch from a black and white image was used as the background for the digit image
    * MDRBI: MNIST digits with rotation and background image - The perturbations used in MRD and MBI were combined.
    * MRB: MNIST random background - A random background was inserted in the digit image
    * MRD: MNIST rotated digits - The digits were rotated by an angle generated uniformly between 0 and 360 radians.
    
* CONVEX dataset - 8000 train, 50000 test

## Usage of the package

### download the datasets

Download the datasets and put the files under the root directory of your project as shown in the following picture. 

![alt text](https://github.com/wwwbbb8510/bidcap/blob/master/dataset_file_structure.PNG "Datasets file structure")

### Load the datasets
```python
# import the loader tool
from bidcap.utils.loader import ImagesetLoader
# import mb. Pass the dataset name described above as the first parameter
data = ImagesetLoader.load('mb')
# training images
data.train['images']
# training labels
data.train['labels']
# test images
data.test['images']
# test labels
data.test['labels']
```
