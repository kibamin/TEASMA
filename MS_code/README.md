
# MS and FDR

## Getting Started

### Environment Setup

- **Create a Virtual Environment:**
    ```bash
    python -m venv MS_env
    ```
- **Activate the Virtual Environment:**
    ```bash
    source MS_env/bin/activate
    ```
- **Install Required Packages:**
    ```bash
    pip install -r requirements_MS.txt
    ```

### Step 1: Generate Mutants

To replicate the results of this project, follow these steps to generate mutants for each subject:

#### A. Train the Original Model

Start by training the original model using the following command:

```bash
python training_models/train.py
```

This command will initiate the training process and create a models folder containing the trained models for each subject.

* Note: Alternatively, you have the option to use the pre-trained models from our experiments. To do this, follow these steps:

1) Download the models.zip file from the replication package.
2) Extract the contents of models.zip to the MS_code directory path of the project.

**B. Generate the mutants**

To generate mutants for each subject, execute the following command:

```bash
python cnn_mutation/src/generator.py --model_path models/{subject_name}/original_model.h5 --subject_name {subject_name} --data_type {dataset_name} --threshold 0.9 --operator -1 --ratio 0.01 --save_path mutants/{subject_name} --num 50
```


The `{subject_name}` and `{dataset_name}` parameters should be set with values from the tables below, which have been used in this project:


| ID | subject_name  | dataset_name  |
|----| -------------|:-------------:|
| S1 | lenet5_mnist        | mnist         |
| S2 | cifar10       | cifar10       |
| S3 | resnet20_cifar10      | cifar10       |
| S4 | lenet4        | fashion_mnist |
| S5 | lenet5_SVHN   | SVHN |
| S6 | vgg16_SVHN        | SVHN |
| S7 | resnet152_cifar100        | cifar100 |
| S8 | inceptionV3_imagenet        | imagenet |

Additionally, the `operator` parameter in the script determines the type of operator applied to the original model to generate mutants. Use the operators outlined in the table below:

| param operator  | description|  
|----|:-------------: |
| -1 | Apply All Operators        | 
| 0  | GF      | 
| 1  | WS     | 
| 2  | NEB    | 
| 3  | NAI    | 
| 4  | NS     | 
| 5  | LR     | 
| 6  | LA     | 
| 7  | LD     | 

The above script will generate mutants for each subject and save them to the `mutants` folder and also generate `mutant_prediction_outputs` that are output probability of execution original training set.


##### Using Pre-Generated Mutants
If you wish to use the mutants generated from our experiment, you can download the `mutants.tar.gz` file and extract its contents to the root of your project and download `mutant_prediction_outputs.zip` file and extract into the mutants directory. In this case, you can skip the next step (Finding Equivalent Mutants).


**Note**: Please be aware that due to the large size of mutants (60GB) for subject 6 (vgg16_SVHN) and subject 8 (inceptionV3_imagenet), we were unable to upload them.


**C. Finding Equivalent Mutants**

Once you have generated mutants for a subject, proceed to identify equivalent mutants using the following script:

```
python training_models/mutant_filtering.py --subject {subject_name}
```
Executing this command will generate an array containing the equivalent mutants, which can be found in the `mutants/equivalent_mutants` folder.








   
### Step 2. Generate Subset Sampling of Training set and Test set

In this phase of the project, datasets will be acquired automatically using the `keras` library, except for the `SVHN` dataset and `imagenet`. To acquire the SVHN dataset, you can conveniently download it from [The Street View House Numbers](http://ufldl.stanford.edu/housenumbers/). You will need to download the following files:

Download [training set](http://ufldl.stanford.edu/housenumbers/train_32x32.mat)
Download [test set](http://ufldl.stanford.edu/housenumbers/test_32x32.mat)

And for `imagenet` you have to do download the dataset manually and put them into datasets directorie (before download this dataset please see the `training_models/imagenet_evaluation.py`)

After downloading these files, organize your project directory as shown below:

        root
        │
        └── datasets
            │
            └── SVHN
                │
                └── train_32x32.mat
                └── test_32x32.mat



Next, execute the following script to generate subset samples from both the training set and the test set:

Execute the below script to produce all subsets of each dataset:
This command generates `data` folder which include all subsets with different size that sampled from training set and test set:

```
python test_suite_handler.py
```
Note: we generated all subsets for our experiments, but they are so large that we could not upload them here.


Running this command will create a `data` folder containing various subsets of different sizes sampled from both the training set and the test set that test is for evaluation time.

**Note**: While we have generated all the subsets for our experiments, the large size of these subsets prevents us from uploading them here. However, using the provided instructions, it is possible to generate not identical but similar sample subsets.





### Step 3. Mutation Score (MS)

In this step, we will demonstrate how to execute mutants on subsets for each subject. This process will help us calculate the Mutation Score (MS) for each subject. Follow the instructions below:

##### * Execute on training subsets
Use the following command to calculate MS for each subject on training subsets:

```
python cnn_mutation/src/runner.py --experiment 0 --test_suite_size {subset_size} --uniform_sampling False --dataset {dataset_name} --model_name {subject_name} --sampling_from training_set

```
Replace {subset_size} with the desired training set sampling size from the available options: 100, 300, 500, 1000, 1500, 3000, 5000, 8000, 12000, 16000, 20000, 24000, ...


##### * Execute on test subsets (for evaluation)
To calculate MS on test subsets, use the following commands:

First execute the bellow command just one time for each subject to (If you already downloaded the `mutant_prediction_outputs.zip` and extracted it skip this step):
```
python training_models/mutant_testing.py --subject {subject_name} --dataset {dataset_name} --dataset_part test_set

```

Then for eache subject and subset size execuate this command:
```
python cnn_mutation/src/runner.py --experiment 0 --test_suite_size {subset_size} --uniform_sampling False --dataset {dataset_name} --model_name {subject_name} 

```
use param {subset_size} for test set sampling: 100, 300, 500, 1000, 1500, 3000, 5000, 8000, ...


Replace {subset_size} with the desired test set sampling size from the available options according to generated subset sizes.

Running these scripts will provide the MS values for experiments E1, E2, and E3. The output will be stored in the all_results folder at the root of the project.

**Note**: If you already have an all_results folder, make sure to rename it before running these scripts to avoid overwriting existing data.

    MS_code
    │
    └── all_results
        │
        └── {subject_name}
            │
            └── 0.01 (ratio)
                │
                └── non_uniform_sampling
                |   │
                |   └── bs_{subset_size}
                |       │
                |       └── bs_{subset_size}_result_E1_E2_E3.csv
                |
                └── training_set
                    │
                    └── non_uniform_sampling
                        │
                        └── bs_{subset_size}
                            │
                            └── bs_{subset_size}_result_E1_E2_E3.csv






### Step 4. Fault Detection Rate (FDR)
This step focuses on estimating faults and subsequently calculating the Fault Detection Rate (FDR).

#### Fault extraction
##### - Find misprediction and feature extraction
Execute the following script to identify mispredictions of each subject and subsequently extract their features:

```
python fault_extraction/mis_predicted_feature_extraction.py --subject {subject_name} --dataset_name {dataset_name}
```

* For subject 8 (inceptionV3_imagenet) use the following script to extract the features of misprediction inputs
```
python fault_extraction/mis_predicted_feature_extraction_for_imagenet.py
```


##### - Fault estimation 
To estimate the fault we used the clustering method to identify faults introduced by [Aghababaeyan et al.](https://arxiv.org/abs/2112.12591)

- Create virtual environment

    `python -m venv hdbscan_env`
- Activate virtual environment

    `source hdbscan_env/bin/activate`
- Install the required packages using the `requirements_hdbscan.txt` file.

    `pip install -r requirements_hdbscan.txt`


To identify faults for each subject, execute the following scripts:


```
python fault_extraction/fault_clustering_estimation.py --model_name {subject_name} --source_data_for_clustering training
```

These scripts will generate an output folder containing information about mispredictions and faults. (**Note**: If the output folder already exists, please rename or remove it to prevent overwriting.)


#### FDR calculation

Execute the below script to compute the Fault Detection Rate (FDR) for each subject on both subsets of the training set and the test set:

```
python FDR_calculation/FDR_score_calculation.py --subject {subject_name}
```

Running this script will generate a `FDR_output` directory with the following structure:


           root
             │
             └── FDR_calculation
                 │
                 └── `FDR_output`
                     │
                     └── training_set
                     |   │
                     |   └── {subject_name}
                     |       │
                     |       └── sampled_test_suites
                     |          │
                     |          └── bs_{size}/result.csv
                     |
                     └── test_set
                         │
                         └── {subject_name}
                             │
                             └── sampled_test_suites
                                │
                                └── bs_{size}/result.csv


**Note**: You can execute steps 3 and 4 concurrently. The order of execution does not impact the results.








### Step 5. Correlation and regression analysis between MS and FDR
At this stage, the Mutation Score (MS) and Fault Detection Rate (FDR) data are prepared for each subject and are ready for analysis. As previously outlined in Quick Replication, you can generate the experimental results by executing the following script. The generated results will be stored in the `plots/correlation_outputs/training_set` folder.

```
python plots/correlation_build_model_MS_FDR.py --subject {subject_name}
```

Upon running this script, you will find the generated figures and tables, consistent with the visualizations and data presented in the paper.

This step enables you to explore and understand the correlation and regression analysis between Mutation Score (MS) and Fault Detection Rate (FDR) for each subject, contributing to a deeper insight into the experiment's findings.

## Important: Ensure you use the same FDR_output of each subject for other metrics (IDC and SC).