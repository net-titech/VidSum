# Research progress

Updated: 2016/12/07

## Milestones

### 1. Dec 7th to Dec 21st

1. Draft paper:
  1. Problem definition.
  2. Contributions.
  3. Comparision to existing algorithms.
    - Random
    - K-Means Centroid
    - K-Means Stillness
    - G-Lasso
    - Beauty Rank
    - CNN
    - ATS Supervised (Song, CIKM 2016)
    - ATS Unsupervised (Song, CIKM 2016)
    - emDPP (Gillenwater, NIPS 2014)
    - SeqDPP (Gong, NIPS 2014)
    - VSumm (de Avila, Pattern Recognition Letter 2011)
  4. Download paper format (maybe CVPR).
  5. List datasets and experiments (Mocap - CMU).
2. Dataset:
  1. Download datasets 
    - Basketball, Boxing, and General Exercises & Stretching (Mocap - CMU).
    - OpenVideo Project (OVP) used by (Gong, NIPS 2014)
    - YouTube (VSUMM) used by (Gong, NIPS 2014)
3. Packing other codes:
  1. Collect the source code of other methods.
    - em-DPP
    - seqDPP
    - FLID
    - VSumm
  2. Design common interface for video summarization.
  3. Pack algorithms into portable Python packages.
4. Collect more related paper on video summarization.

### 2. Dec 21st to Jan 3rd

1. Running all baseline methods using all the datasets
  1. Baseline Methods:
    - seqDPP Linear
    - seqDPP NNets
    - VSumm1
    - VSumm2
    - DT
    - OV 
    - STIMO
  2. Dataset:
    - OpenVideo Project (OVP) 
    - YouTube 
2. Modify FLID for processing the same input and output as baseline methods
3. Implementing Theano + Keras
4. Find more related paper for evaluation:
  1. Diversity evaluation
  2. Non expert/user judgment
  

## Resources
1. [VSum](https://sites.google.com/site/vsummsite)
2. [seqDPP](https://github.com/pujols/Video-summarization)
3. [emDPP](https://code.google.com/archive/p/em-for-dpps)
4. [Yahoo Hecate](https://github.com/yahoo/hecate)
5. [Thumbnail](https://github.com/yalesong/thumbnail)
6. [TV Sum](https://github.com/yalesong/tvsum)


## Possible Target
1. [IEEE ICIP 2017](http://2017.ieeeicip.org/), Jan 31, 2017
2. [ACM ICMR 2017](http://www.icmr2017.ro/), Jan 27, 2017
3. [ICCV 2017](http://iccv2017.thecvf.com/), March 10, 2017
