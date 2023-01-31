# WeiRules
Source code for the neuro-fuzzy network of the paper: "Logic Rules Meet Deep Learning: A Novel Approach for Ship Type Classification"

## Description
WeiRules is a neuro fuzzy network that leverages logic rules on tabular data and deep features corresponding to bounding boxes in images for classification on attribute correlated images. In detail, the logic rules on the tabular data attributes are extracted from decision tree(s), then fuzzified using the sigmoidal membership function and exponential mean approximations for conjunction and (weighted) disjunction. The fuzzified rules are integrated into a neural network that takes two inputs: (a) the tabular data fields, (b) deep features extracted from a pretrained Faster R-CNN model on the images. During the training step of the WeiRules model, the weights of the weighted disjucntion (which are integrated into the network architecture are learned). The output of the model yields a vector with the output of each rule, and therefore the class prediction.

## Example dataset
Since the original maritime dataset includes propriatery images, we demonstrate the usage of weirules on a dataset containing yahoo images correlated with attribute data. The original dataset can be found in [https://vision.cs.uiuc.edu/attributes/](https://vision.cs.uiuc.edu/attributes/)

## Requirements 
  - pytorch                   1.13.1         
  - pytorch-cuda              11.7               
  - torchvision               0.14.1    
  - scikit-learn              1.0.2
  - pandas                    1.4.2
  - jupyterlab                3.5.2
  
## About 
For more information please see the publication below.

Pitsikalis, M., Do, TT., Lisitsa, A., Luo, S. (2021). Logic Rules Meet Deep Learning: A Novel Approach for Ship Type Classification. In: Moschoyiannis, S., Peñaloza, R., Vanthienen, J., Soylu, A., Roman, D. (eds) Rules and Reasoning. RuleML+RR 2021. Lecture Notes in Computer Science(), vol 12851. Springer, Cham. https://doi.org/10.1007/978-3-030-91167-6_14

## Licence
Unless stated otherwise in the provided source files, the included source files are licenced under GPLv3.
