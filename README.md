
---   
<div align="center">    
 
# Adversarial Transfer Learning with Noise Contrastive Model for Machine Remaining Useful Life Estimation     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)

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
git clone https://github.com/YourGithubName/Your-project-name   

# install project   
cd Your-project-name 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to [Your Main Contribution (MNIST here)] and run it.   
 ```bash
# module folder
cd research_seed/mnist/   

# run module (example: mnist as your main contribution)   
python mnist_trainer.py    
```

## Main Contribution      
List your modules here. Each module contains all code for a full system including how to run instructions.   
- [MNIST](https://github.com/williamFalcon/pytorch-lightning-conference-seed/tree/master/research_seed/mnist)  

## Baselines    
List your baselines here.   
- [MNIST_baseline](https://github.com/williamFalcon/pytorch-lightning-conference-seed/tree/master/research_seed/baselines/mnist_baseline)  

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
