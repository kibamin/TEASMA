# DSC and LSC Calculation Guide

To calculate the DSC and LSC of subsets, TEASMA uses the official implementation provided by the Surprise Adequacy Framework [Evaluating Surprise Adequacy for Deep Learning System Testing](https://github.com/less-lab-uva/InputDistributionCoverage/tree/main).


## Getting Started

### Setting Up the Environment

- **Create and activate a virtual environment for SC:**
    ```bash
    python -m venv SC_env
    source SC_env/bin/activate
    ```

- **Install the required dependencies:**
    ```bash
    pip install -r requirements_SC.txt
    ```


### Step 1: compute ATS and SA (LSA and DSA)
To calculate SA (DSA and LSA) for each subject, execute the following scripts:

To calculate DSA
```
python run.py -d {dataset_name} --subject {subject_name} -dsa
```

To calculate LSA
```
python run.py -d {dataset_name} --subject {subject_name} -lsa
```

After executing these scripts, you will have the SA results in `SA_results` that include the necessary files and directories to calculate the Surprise Coverage (SC) of subsets.

* #### For ImageNet dataset we used dnn-tip [Weiss implementation](https://github.com/testingautomated-usi/dnn-tip) to calculate DSA and LSA to reduce running time cost.

### Step 1.1 (DSA and LSA for imagenet)

Execute the following scripts to calculate ATS of training set and Validation set of ImageNet (make sure you download imagenet dataset if not please check the `../MS_code/README.md` and `../MS_code/training_models/imagenet_evaluation.py`):

Run the scripts below to calculate the Average Training Set (ATS) for both the training and validation sets of ImageNet. Ensure you have the ImageNet dataset downloaded; if not, please refer to the README.md located at `../MS_code/` and the `imagenet_evaluation.py` script at `../MS_code/training_models/`.

```
python get_imagenet_activation_prediction.py
```

To calculate DSA for ImageNet:
```
python sa_lib.py -lsa --sampling_from test_set
python sa_lib.py -lsa --sampling_from training_set

```

To calculate DSA for ImageNet:
```
python sa_lib.py -dsa --sampling_from test_set
python sa_lib.py -dsa --sampling_from training_set

```


### Step 2: Train and Test Subsets

Use the same method outlined in the `MS_code` directory to generate subsets. For consistent comparisons between MS and IDC metrics, utilize the same subsets generated for the MS metric. Copy the contents from `data_subsets` (after downloading and extracting them) into a directory within `SC_code/data`. This ensures that both metrics are evaluated under similar conditions. The directory structure should be as follows:

```

SC_code
    │
    └── data
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


### Step 3: Calculate SC for subsets based on DSA and LSA

To calculate SC of each subset you have to execute the following script for all sizes seperatly (Note that we used same configuration of original implemention):

* For training subsets
```
python sc_runner.py  --dataset {dataset_name} --subject {subject_name} --sampling_from training_set -lsa -dsa --test_suite_size {subset_size}


E.g. : python sc_runner.py  --dataset imagenet --subject inceptionV3_imagenet --sampling_from training_set -lsa -dsa --test_suite_size 35000

```

* For test subsets
```
python sc_runner.py  --dataset {dataset} --subject {subject_name} -lsa -dsa --test_suite_size {subset_size}


E.g. : python sc_runner.py  --dataset imagenet --subject inceptionV3_imagenet -lsa -dsa --test_suite_size 8000


```


Running these scripts will provide the SC values for each subset. The output will be stored in the all_results folder at the root (SC_code) of the project. Finally the end you have to see the following structure:

```

SC_code
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

### Step 4: Calculate correlation between SC and FDR and build regression models (RQ1, RQ2, and RQ3)

To calculate the FDR, refer to the instructions in `../MS_code/README.md` or use the FDR outputs located in `../MS_code/FDR_calculation/FDR_output`.

At this stage, the SC and FDR results for each subject are prepared and ready for analysis.

Run the following script to generate the figures and tables consistent with those presented in the paper:

```
python plots/correlation_build_model_SC_FDR.py --subject {subject_name}
```
