# IDC Calculation

The IDC (Input Distribution Coverage) framework assesses the coverage of input distributions for neural network testing. This section details the process of calculating IDC using the official implementation from [InputDistributionCoverage](https://github.com/less-lab-uva/InputDistributionCoverage/tree/main).

## Getting Started

### Setting Up the Environment

- **Create and activate a virtual environment:**
    ```bash
    python -m venv IDC_env
    source IDC_env/bin/activate
    ```

- **Install the required dependencies:**
    ```bash
    pip install -r requirements_IDC.txt
    ```

### Step 1: Train Variational Autoencoder (VAE)
IDC uses a VAE model for each dataset to calculate coverage based on latent vector outputs. We utilize the [Disentangling VAE](https://github.com/YannDubs/disentangling-vae) implementation. Our trained VAE models are stored in the `IDC_code/results` directory.


| dataset | VAE model| latent Z |  
|----|:-------------: | :-------------: |
| MNIST | $\beta$-TCVAE        | 6 |
| Fashion_MNIST  | $\beta$-TCVAE      | 6 |
| CIFAR-10  | $\beta$-TCVAE      | 32 |
| CIFA100  | $\beta$-TCVAE     | 32 |
| SVHN  | $\beta$-TCVAE    | 64 |
| ImageNet  | factor    | 64 |


### Step 2: Train and Test Subsets
Use the same method outlined in the `MS_code` directory to generate subsets. For consistent comparisons between MS and IDC metrics, utilize the same subsets generated for the MS metric. Copy the contents from `data_subsets` (after downloading and extracting them) into a directory within `IDC_code/data_subsets`. This ensures that both metrics are evaluated under similar conditions. The directory structure should be as follows:


```

IDC_code
    │
    └── data_subsets
        │
        └── {dataset_name}
            │
            └── sampled_test_suits
            |   │
            |   └── batch_size_{subset_size}
            |       │
            |       └── test_suites_all.csv
            |
            └── training_set
                │
                └── sampled_test_suits
                    │
                    └── batch_size_{subset_size}
                        │
                        └── test_suites_all.csv


```


### Step 3: Calculate IDC for subsets

To calculate IDC of each subset you have to execute the following script for all sizes seperatly:

* For training subsets
```
python measure_coverage_for_subsets.py  {vae model name} --dataset {dataset_name} --sampling_from training_set --test_suite_size {subset_size}

E.g. : python measure_coverage_for_subsets.py  btcvae_cifar100_z32_e300 --dataset cifar100 --sampling_from training_set --test_suite_size 100


or


xvfb-run -a python measure_coverage_for_subsets.py  {vae model name} --dataset {dataset_name} --sampling_from training_set --test_suite_size {subset_size}

```

* For test subsets
```
python measure_coverage_for_subsets.py  {vae model name} --dataset {dataset_name} --test_suite_size {subset_size}

E.g. : python measure_coverage_for_subsets.py  btcvae_cifar100_z32_e300 --dataset cifar100 --test_suite_size 100


or


xvfb-run -a python measure_coverage_for_subsets.py  {vae model name} --dataset {dataset_name} --test_suite_size {subset_size}

```


Running these scripts will provide the IDC values for each subset. The output will be stored in the all_results folder at the root (IDC_code) of the project. You have to copy the result of each dataset to each subject directory related to the same dataset. for example for subject `lenet5_SVHN` and `vgg16_SVHN` you have to make two folder with same subject name and copy `all_results/SVHN` contents into `lenet5_SVHN` and `vgg16_SVHN`. At the end you have to see the following structure:

Executing these scripts will generate the IDC values for each subset. The results will be stored in the `all_results` folder within the `IDC_code` directory at the project's root. For each dataset, you should distribute the results into the respective subject directories. For instance, for subjects `lenet5_SVHN` and `vgg16_SVHN`, create two folders named after each subject and copy the contents from `all_results/SVHN` into both `lenet5_SVHN` and `vgg16_SVHN`. The final directory structure should appear as follows:


```

IDC_code
    │
    └── all_results
        │
        └── {subject_name}
            |
            └── non_uniform_sampling
            |   │
            |   └── bs_{subset_size}
            |       │
            |       └── bs_{subset_size}_result_SC.csv
            |
            └── training_set
                │
                └── non_uniform_sampling
                    │
                    └── bs_{subset_size}
                        │
                        └── bs_{subset_size}_result_SC.csv

```

### Step 4: Calculate correlation between IDC and FDR and build regression models (RQ1, RQ2, and RQ3)

To calculate the FDR, refer to the instructions in `../MS_code/README.md` or use the FDR outputs located in `../MS_code/FDR_calculation/FDR_output`.

At this stage, the IDC and FDR results for each subject are prepared and ready for analysis.

Run the following script to generate the figures and tables consistent with those presented in the paper:

```
python plots/correlation_build_model_IDC_FDR.py --subject {subject_name}
```