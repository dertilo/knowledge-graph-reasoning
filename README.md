# knowledge-graph-reasoning
## thoughts
* multihopkg/ConvE approach is fully sufficient; no negative sampling needed, always score against all entities, might not work for graphs with millions of entities
### attempt to build a pytorchic KG-embedding+reasoning system; inspired by the following ones 
#### 1. [MultiHopKG](https://github.com/salesforce/MultiHopKG)
* during training of ConvE scores against all entities ([see](https://github.com/salesforce/MultiHopKG/blob/23747699aefe3a9f835f9875ce446a18de96dcb1/src/emb/fact_network.py#L142)) I guess this only works for very small graphs

##### pros: 
* nice collection of [benchmark-datasets](https://github.com/salesforce/MultiHopKG/blob/master/data-release.tgz)
* implements multihop reasoning (powered by reinforcement learning)
##### cons: 
* use of [inheritance](https://github.com/salesforce/MultiHopKG/blob/23747699aefe3a9f835f9875ce446a18de96dcb1/src/emb/emb.py#L21)
* no division of concerns ([see](https://github.com/salesforce/MultiHopKG/blob/23747699aefe3a9f835f9875ce446a18de96dcb1/src/learn_framework.py#L291)) 
* not actively developed
#### 2. [PyKEEN](https://github.com/SmartDataAnalytics/PyKEEN)
##### pros: 
* many algorithms implemented
##### cons: 
* unnecessary obfuscation+complexity ([see](https://github.com/SmartDataAnalytics/PyKEEN/blob/d420aed7cd10fc883b70fcd4c920e8edec7fb6ce/src/pykeen/utilities/pipeline.py#L30))
* not actively developed (?)

#### 3. [SimplE](https://github.com/baharefatemi/SimplE)
##### pros: 
* simplicity
##### cons: 
* not actively developed

#### 4. [OpenKE](https://github.com/thunlp/OpenKE)
##### pros: 
* __actively developed__!
##### cons: 
* use of [cpp-code](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/openke/base)

# Datasets

* WN18RR is shit? -> [see](https://github.com/villmow/datasets_knowledge_embedding)

[CogKR](https://github.com/THUDM/CogKR)
[AmpliGraph](https://github.com/Accenture/AmpliGraph)