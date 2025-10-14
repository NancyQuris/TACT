# Test-Time Adaptation by Causal Trimming 
This repository contains the implementation of the TACT algorithm and empirical studies in Test-Time Adaptation by Causal Trimming in NeurIPS 2025. 
The full paper can be found at https://arxiv.org/abs/2510.11133. 

## Overview
A primary cause of performance degradation under distribution shifts is the model’s reliance on features that lack a direct causal relationship with the prediction targets. 
Building on prior studies that suggest representations learned through standard training encode a mixture of causal and non-causal features, the causal part is often learned sufficiently well and high-level semantics are often linearly encoded in the learned representation space, TACT identifies and removes non-causal components from representations for test distributions. 

To identify non-causal features, TACT applies data augmentations that preserve causal features while varying non-causal ones on the test samples.
By analyzing the changes in the representations under such augmentation using Principal Component Analysis, TACT identifies the highest variance directions associated with non-causal features.
To reduce non-causal features, TACT trims the representations by removing their projections on the identified directions, and uses the trimmed representations for the predictions. 
Since the prototypes used for prediction, i.e. weights for each class in the linear classifier, are also influenced by non-causal features, we apply the same operation to them using the identified non-causal direction. During adaptation, we maintain a moving average of the updated prototypes to mitigate noise effects.

On benchmark datasets, we achieve the following results compared with other backpropagation-free methods:

| Method        | Birdcalls  | Camelyon17 | CivilComments | ImageNet-R | ImageNet-V2 |
|---------------|------------|------------|---------------|------------|-------------|
|*No TTA*       | 22.74      | 62.31      | 55.38         | 41.83      | 62.97       |
| T3A           | 26.16±1.33 | 69.96±1.98 | 56.43±0.00    | 41.78±0.12 | 62.93±0.02  |
| LAME          | 23.66±1.01 | 62.38±0.03 | 56.24±0.10    | 41.77±0.01 | 63.00±0.02  |
| FOA           | 26.95±1.81 | 58.36±0.77 | -             | 41.46±0.16 | 62.76±0.08  |
| TACT          | 31.14±1.69 | 70.17±0.05 | 71.80±0.35    | 43.59±0.02 | 63.33±0.10  |

We also implement a variant called TACT-adapt, where predictions from TACT are used to guide gradient-based model updates with cross entropy loss, regularized by information maximization loss. we achieve the following results compared with other backpropagation-based methods:

| Method     | Birdcalls  | Camelyon17 | CivilComments | ImageNet-R | ImageNet-V2 |
|------------|------------|------------|---------------|------------|-------------|
|*No TTA*    | 22.74      | 62.31      | 55.38         | 41.83      | 62.97       |
| SHOT       | 26.82±5.14 | 80.28±5.61 | 13.93±0.97    | 48.79±0.08 | 63.32±0.09  |
| Tent       | 23.16±0.42 | 62.29±0.01 | 55.38±0.00    | 42.08±0.05 | 63.09±0.03  |
| SAR        | 23.16±0.42 | 62.30±0.00 | 55.38±0.00    | 42.58±0.11 | 62.97±0.01  |
| DeYO       | 23.29±0.39 | 69.64±1.47 | -             | 46.87±0.08 | 62.96±0.01  |
| TAST       | 26.08±1.11 | 83.01±1.42 | 56.56±0.20    | 41.09±0.08 | 62.84±0.07  |
| TSD        | 27.33±1.75 | 67.33±4.74 | 55.38±0.00    | 41.76±0.01 | 62.98±0.01  |
| PASLE      | 27.35±1.79 | 60.66±0.04 | 55.77±0.15    | 46.08±0.09 | 63.15±0.04  |
| TACT-adapt | 31.25±3.59 | 83.70±1.10 | 71.98±0.19    | 48.81±0.05 | 63.44±0.07  |

## Implementation
### Requirements 
The packages required are specified in `requirements.txt`.
Please run `pip install -r requirements.txt` to install them.

To prepare for the datasets, 
- Birdcalls, Camelyon17 and CivilComments will be downloaded during adaptation if the datasets are not found in the specified data directory. 
- ImageNet-R can be downloaded via this [link](https://github.com/hendrycks/imagenet-r).
- ImageNet-V2 will be downloaded via the dependencies installation mentioned above.

### Preparing Models for Adaptation
We include the models used for Birdcalls and Camelyon in the release. `main.py` was used to train the models. 

The model used for CivilComments can be downloaded via this [link](https://worksheets.codalab.org/rest/bundles/0x17807ae09e364ec3b2680d71ca3d9623/contents/blob/best_model.pth).

The model used for ImageNet-R and ImageNet-V2 will be downloaded by torchvision during adaptation if the pretrained weight is not available.

### Test-Time Adaptation 

To perform test-time adaptation on Birdcalls and Camelyon17, run 
```
    python test_time_adapt_wilds.py \
        --dataset dataset_to_adapt \
        --data_dir /path/to/data_dir/ \
        --eval_batch_size test_batch_size \
        --model_path /path/to/model \
        --algorithm TACT
        
```

To perform test-time adapation on CivilComments, run 
```
    python test_time_adapt_wilds.py \
        --dataset civil \
        --data_dir /path/to/data_dir/ \
        --eval_batch_size test_batch_size \
        --use_published_model \
        --model_path /path/to/model \
        --algorithm TACT
```

To perform test-time adaptation on ImageNet-R and ImageNet-V2, run
```
    python test_time_adapt_imagenet.py \
        --dataset dataset_to_adapt \
        --data_dir /path/to/data_dir/ \
        --eval_batch_size test_batch_size \
        --algorithm TACT
``` 

### Quick hyperparameter search for TACT
Instead of generating augmented data each time for every `num_aug` and `num_pcs` and performing eigendecomposition for every `num_pcs` when `num_aug` is the same, we implement a quick hyperparameter search version for TACT where the augmented data and eigenvectors are saved for subsequent hyperparameter search. 

For Birdcalls and Camelyon17, run 
```
    python -m quick_hparam_search.wilds \
        --dataset dataset_to_adapt \
        --data_dir /path/to/data_dir/ \
        --eval_batch_size test_batch_size \
        --model_path /path/to/model
        
``` 

For CivilComments, run 
```
    python -m quick_hparam_search.wilds \
        --dataset civil \
        --data_dir /path/to/data_dir/ \
        --eval_batch_size test_batch_size \
        --use_published_model \
        --model_path /path/to/model 
```

For ImageNet-R and ImageNet-V2, run
```
    python -m quick_hparam_search.imagenet \
        --dataset dataset_to_adapt \
        --data_dir /path/to/data_dir/ \
        --eval_batch_size test_batch_size
``` 

### GradCAM Visualization
`visualization.ipynb` contain the code to visualize original predictions and predictions made by TACT. 

<!-- ## Citation 
If you find our paper useful, you are welcome to cite it as
```
``` -->