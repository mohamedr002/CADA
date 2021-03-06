
---   
<div align="center">    
 
# Contrastive Adversarial Domain Adaptation for Machine Remaining Useful Life Predictionn     

[![Paper](http://img.shields.io/badge/IEEE Transactions on Industrial Informatics.1001.2234-B31B1B.svg)](https://ieeexplore.ieee.org/document/9234721)

<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->



<!--  
Conference   
-->   
</div>
 
## Description   
In this paper we propose adversarial transfer domain adaptation approach with noise contrastive estimation approach for remaining useful life estimation of turbofan engine. The proposed approach aim to align the feature distrubtion between the source and target using adversarial approach. However, this approach can remove target specific information to align the two distribution, which can deteriorate the performance. To handle this issue, we propose to use noise contrastive estimation approach to maximize the mutual information between the target input and target features to keep the semantic strucutre of the target data during domain alignmet. 
## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/mohamedr002/ATL_NCE  

# install project   
cd ATL_NCE   
pip install -r requirements.txt
 ```   
 Next, Downlaod the turbofan engine dataset from here  [CMAPPS](https://catalog.data.gov/dataset/c-mapss-aircraft-engine-simulator-data)
  ```bash
# run_the data/data_preprocessing.py to apply the preprocessings.
```
 Finally, run the code to expirement the model among different datasets. 
 ```bash
# run module (example: mnist as your main contribution)   
python main_cross_domains.py    
```

## Main Contribution      


## Baselines    

### Citation   
```
@article{Mohamed Ragab,
  title={Adverarial Transfer Learning with Noise Constrastive for Machine Remaining Useful Life estimation },
  author={Your team},
  journal={Location},
  year={Year}
}
```   
