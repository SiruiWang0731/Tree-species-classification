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

## Abstract

Tree species classification plays an important role in the fields of ecosystem monitoring and biomass prediction. Machine learning (ML) for Remote Sensing (RS) method provides an efficient way to classify tree species with low cost and high accuracy. To not only get information about location and extent but also about the species distribution high accuracy labeled tree species data is needed. In this project work the possibilities of the EU-Forest dataset [Mauri et al., 2017] were tested for tree species classification using a machine learning approach.  ESA’s Sentinel-2 data was used as the main dataset for the analysis. After cloud masking, three season-stacks have been generated. Vegetation indices have been calculated as extra features and were added to the stacks. 5 tree species were classified by Random Forest (RF), Artificial Neural Network (ANN), Convolutional Neural Network (CNN), Residual Neural Network (ResNet) [He et al., 2015] and Recurrent Neural Network (RNN) [Mou et al., 2017]. Based on the high-performing models, a decision fusion model was built. The Recurrent Residual Convolutional Neural Network (RRCNN) with CNN, Resnet, and RNN. The final overall accuracy varies from 49.352% (ANN) to 73.33% (Resnet 50) for the validation but lies only at 20.40% for the testing (RNN).

![This is an image](https://github.com/SiruiWang0731/Tree-species-classification/blob/5f27eaaba9bf5d29d07ead18b8e9375f54587a28/Screenshot%202023-02-08%20at%2023.18.48.png)

## References:

[1] Tree species introduction: https://www.tree-guide.com/

[2] Kuenzer,  C.;  Ottinger,  M.;  Wegmann,  M.;  Guo,  H.;  Wang,  C.;  Zhang,  J.;  Dech,  S.;  Wikelski,  M. (2014).  Earth observation satellite sensors for biodiversity monitoring: Potentials and bottlenecks.Int. J. Remote Sens, 35, 6599–6647

[3] Immitzer, M., Atzberger, C., & Koukal, T. (2012). Tree species classification with Random forest using very high spatial resolution 8-band worldView-2 satellite data. Remote Sensing, 4(9), 2661–2693. https://doi.org/10.3390/rs4092661

[4] Immitzer, M., Neuwirth, M., Böck, S., Brenner, H., Vuolo, F., & Atzberger, C. (2019). Optimal input features for tree species classification in Central Europe based on multi-temporal Sentinel-2 data. Remote Sensing, 11(22). https://doi.org/10.3390/rs11222599

[5] Dalponte, M., Ørka, H. O., Gobakken, T., Gianelle, D., & Næsset, E. (2013). Tree species classification in boreal forests with hyperspectral data. IEEE Transactions on Geoscience and Remote Sensing, 51(5), 2632–2645. https://doi.org/10.1109/TGRS.2012.2216272

[6] Mauri, A., Strona, G., & San-Miguel-Ayanz, J. (2017). EU-Forest, a high-resolution tree occurrence dataset for Europe. Scientific data, 4(1), 1-8.

[7] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. http://image-net.org/challenges/LSVRC/2015/

[8] Hoffmann, E. J., Wang, Y., Werner, M., Kang, J., & Zhu, X. X. (2019). Model fusion for building type classification from aerial and street view images. Remote Sensing, 11(11). https://doi.org/10.3390/rs11111259

[9] Corona, N., Mauricio Galeana-Pizaña, J., Manuel Núñez, J., Luis Silván Cárdenas, J., Corona Romero, N., Mauricio Galeana Pizaña, J., Manuel Nuñez Hernández, J., Manuel Madrigal Gómez, J., Silván Cárdenas, J. L., Corona Romero, N., & Galeana Pizaña, J. M. (2015). GEOSPATIAL TECHNOLOGIES TO SUPPORT CONIFEROUS FORESTS RESEARCH AND CONSERVATION EFFORTS IN MEXICO. https://www.researchgate.net/publication/283503310

[10] Mou, L., Ghamisi, P., & Zhu, X. X. (2017). Deep recurrent neural networks for hyperspectral image classification. IEEE Transactions on Geoscience and Remote Sensing, 55(7), 3639–3655. https://doi.org/10.1109/TGRS.2016.2636241
