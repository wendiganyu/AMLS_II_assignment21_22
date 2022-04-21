# AMLS-II_Final_Project
This is my final project of Applied Machine Learning Systems course II, which does the image super resolution on DIV2K 
dataset using models of SRResnet and SRGAN. SRResnet has the same network structure as the SRGAN generator but a 
different loss function. The results mainly show that SRGAN generates more realistic SR outputs than SRResnet. 
And both models’ performances degrade with the upscale factor growing from 2 to 4.
![image](https://github.com/wendiganyu/AMLS_II_assignment21_22/blob/main/images/Pre.png)

# Project Structure
```bash
/AMLS_II_assignment21_22
│  DataLoad.py: Implements the functions to load DIV2K datasets with PyTorch.
│  Model_GAN.py: Define the model structure of SRGAN with PyTorch.
│  README.md
│  requirements.txt: Python package requirements.
│  Train_Model_GAN.py: Run this file to train the SRGAN model.
│  Train_Model_SRResnet.py: Run this file to train the SRResnet model.
│  Utils.py: Utility functions mainly doing image processing and loading the saved SR generator model.
│  Visualization.py: Generate the visualization results by loading the saved models. The visualization figures are used for report.     
│  RunRecords.zip: My program running records of the training processes of SRResnet and SRGAN.
├─Datasets: Store the train, valid, and testing datasets.
├─images: Store the images of reconstructed pictures, training and validating curves, etc.
├─results: Store the trained model files. Generated antomatically when running the program.
└─summary_writer_records: Records of the training losses, validating metrics, etc. Saved with Tensorboard Summarywriter.
```
The contents Dataset/ aren't contained under this repository, you need to download them manually.

## How to run
First set up the Python environment with required packages. I use the Python version of 3.9.
To install the required packages:
```
pip install -r requirements.txt
```

Then, download the dataset: [Dataset](https://drive.google.com/file/d/1YqX7FZg3DjGmdQs6yDjvuTzz-qKeK_dt/view?usp=sharing). Unzip it and put the contents under Datasets/ folder.

Then, the following are the available commands to train SRResnet or SRGAN, with different operator factors and different upscale factors. Run the command under the path of this project directory:
```
python Train_Model_SRResnet.py --track="BicubicX2" 
python Train_Model_SRResnet.py --track="BicubicX3" 
python Train_Model_SRResnet.py --track="BicubicX4" 
python Train_Model_SRResnet.py --track="UnknownX2" 
python Train_Model_SRResnet.py --track="UnknownX3" 
python Train_Model_SRResnet.py --track="UnknownX4" 

python Train_Model_GAN.py --track="BicubicX2" 
python Train_Model_GAN.py --track="BicubicX3" 
python Train_Model_GAN.py --track="BicubicX4" 
python Train_Model_GAN.py --track="UnknownX2" 
python Train_Model_GAN.py --track="UnknownX3" 
python Train_Model_GAN.py --track="UnknownX4"
```
In interested, you can check my program running records by unzipping RunRecords.zip.

If you want to see the visualization results with the saved model, you can first download and unzip the model files to the results/ folder: [Results](https://drive.google.com/file/d/1l_D4dleZ427yqDYGBXeuuWJ2pVNHL7z1/view?usp=sharing)
And then run:
```
python Visualization.py
```
Or if you are interested of training the model yourself and check the visualization results, you can adjust the corresponding codes indicating the path of the saved models in Visualization.py and run.

### Prerequisites

See requirements.txt
