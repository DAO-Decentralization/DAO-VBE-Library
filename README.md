# DAO-VBE-Library
This repo is a quickstart to easily calculate the Voting-Bloc Entropy (VBE) metric on Optimism data, which showcases how decentralized voting blocs are in an organization (higher VBE = greater decentralization).

In this repo, the below scripts provide the following functions:
- ```load_data.py```: loads and formats the data from RetroPGF voting rounds. Cleans data, removes duplicates, and flags issues.
- ```calculate_vbe.py```: performs clustering for voter feature data, and computes VBE as a function on the size of the largest cluster.
- ```utils.py```: used for supporting functions in loading data, calculating optimal model parameters, and saving data.
- ```results/```: saves report for VBE and model parameters, as well as clustering data.
- ```data/```: where to drop csv file for data loading

## Instructions to run

1. In your Command Line Interface (CLI), ```git clone``` into your desired directory
2. Drop your csv data file into the data/ folder
3. Change directory to ```vbe``` by entering ```cd optimism-dao-vbe/vbe/```
4. Enter ```python calculate_vbe.py ``` to begin
5. Follow the instructions prompted in the interface

To save model parameters and VBE, make sure to enter "Y" when prompted. Alternatively, if you want to save the cluster groupings against the original voter data, make sure to change the default "N" to "Y" when prompted.

### Parameters
Below parameters can be set, including:

- **Clustering Model:** (K-means (default), DBSCAN, Hierarchical Gaussian Mixture Models, Spectral Clustering)
- **Optimization methods:** (Silhouette Score (default), Gap Statistic, Davies-Bouldin, Calinski Harabasz, K-distance, BIC, AIC, Log-Likelihood)
- **Distance Metric for model:** (Euclidean (default), Manhattan, Cosine, Cityblock, L1, L2)
- **Distance Metric for optimization:** (Euclidean (default), Manhattan, Cosine, Cityblock, L1, L2)
- **Data Scaler:** (Min-Max (default), Standard)
- **Entropy Function:** (Min Entropy (default), Max Entropy, Shannon Entropy)
- Entry data path
- Data save path

## Running the model directly
You may run the model directly by entering all relevant parameters using a format like below:
- ```python calculate_vbe.py --model "K-means (default)" --num_clusters 3 --optimization_method "Silhouette Score (default)" --dist_method "Euclidean (default)" --save_clusters "Y" ```
<br />
To view the full information in the command line, you can use ```python calculate_vbe.py --help```

### Model hyperparameters

    parser.add_argument('--model', type=str, default=None, help='Model to use for training')
    parser.add_argument('--num_clusters', type=int, default=2, help='Number of clusters to use')
    parser.add_argument('--dist_model', type=str, default='Euclidean', help='Distance method for clustering')
    parser.add_argument('--optimization_method', type=str, default='Silhouette Score (default)', help='Method to find optimal cluster')
    parser.add_argument('--dist_method', type=str, default='Euclidean (default)', help='Distance method for parameter optimization')
    parser.add_argument('--optimize', type=str, default='Y', help='Flag to use optimal number of clusters calculated')
    parser.add_argument('--entropy', type=str, default='Min Entropy (default)', help='Entropy function for VBE calculation')

### Data hyperparameters
    parser.add_argument('--scale', type=str, default='Min-Max (default)', help='Standard or Min-max scaled data')
    parser.add_argument('--save_data', type=str, default='Y', help='Flag to save parameters and VBE')
    parser.add_argument('--save_clusters', type=str, default='N', help='Flag to save clusters and labels')
    parser.add_argument('--save_cluster_path', type=str, default=os.path.join(os.pardir, 'results', 'cluster_data.csv'), help='Path for saving cluster data')
    parser.add_argument('--path', type=str, default=os.path.join(os.pardir, 'data', 'rpgf3_anon_walletlevel_data.csv'), help='Path for data')

