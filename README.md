# Tree-species-classification

Project package description:

1. Python scripts:
‘tree_classification_datapre.ipynb’
‘tree_classification_NN.ipynb’
‘check.py’
‘data_augmentation_script.ipynb’.

2. GEE code:
URL link with script.

3. Reference data:
‘File1.csv’
‘File2.csv’.

3. Training, validation and testing data:
‘Train_merge.npy’
‘Val_merge.npy’
‘Test_merge.npy’.

![This is an image](https://github.com/SiruiWang0731/Tree-species-classification/blob/58e921832328a3938e346923320466bd411eac2c/Screenshot%202023-02-08%20at%2023.16.17.png)

##Abstract
Tree species classification plays an important role in the fields of ecosystem monitoring and biomass prediction. Machine learning (ML) for Remote Sensing (RS) method provides an efficient way to classify tree species with low cost and high accuracy. To not only get information about location and extent but also about the species distribution high accuracy labeled tree species data is needed. In this project work the possibilities of the EU-Forest dataset [Mauri et al., 2017] were tested for tree species classification using a machine learning approach.  ESA’s Sentinel-2 data was used as the main dataset for the analysis. After cloud masking, three season-stacks have been generated. Vegetation indices have been calculated as extra features and were added to the stacks. 5 tree species were classified by Random Forest (RF), Artificial Neural Network (ANN), Convolutional Neural Network (CNN), Residual Neural Network (ResNet) [He et al., 2015] and Recurrent Neural Network (RNN) [Mou et al., 2017]. Based on the high-performing models, a decision fusion model was built. The Recurrent Residual Convolutional Neural Network (RRCNN) with CNN, Resnet, and RNN. The final overall accuracy varies from 49.352% (ANN) to 73.33% (Resnet 50) for the validation but lies only at 20.40% for the testing (RNN).

![This is an image](https://github.com/SiruiWang0731/Tree-species-classification/blob/5f27eaaba9bf5d29d07ead18b8e9375f54587a28/Screenshot%202023-02-08%20at%2023.18.48.png)
