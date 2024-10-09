import os
import argparse
import numpy as np
import pandas as pd
import inquirer
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from load_data import load_data_rpgf
from utils import save_clusters, save_to_csv, return_optimal_kmeans, return_optimal_hierarchical, return_optimal_gmm, return_optimal_spectral, return_optimal_dbscan

def get_args():
    '''
    Arguments for calculating VBE. You may choose to extend or change these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='Calculate VBE')

    # Model hyperparameters
    parser.add_argument('--model', type=str, default=None, help='Model to use for training')
    parser.add_argument('--num_clusters', type=int, default=2, help='Number of clusters to use')
    parser.add_argument('--dist_model', type=str, default='Euclidean', help='Distance method for clustering')
    parser.add_argument('--optimization_method', type=str, default='Silhouette Score (default)', help='Method to find optimal cluster')
    parser.add_argument('--dist_method', type=str, default='Euclidean', help='Distance method for parameter optimization')
    parser.add_argument('--optimize', type=str, default='Y', help='Flag to use optimal number of clusters calculated')
    parser.add_argument('--entropy', type=str, default='Min Entropy (default)', help='Entropy function for VBE calculation')
    
    # Data hyperparameters
    parser.add_argument('--scale', type=str, default='Min-Max (default)', help='Standard or Min-max scaled data')
    parser.add_argument('--save_data', type=str, default='Y', help='Flag to save parameters and VBE')
    parser.add_argument('--save_clusters', type=str, default='N', help='Flag to save clusters and labels')
    parser.add_argument('--save_cluster_path', type=str, default=os.path.join(os.pardir, 'results', 'cluster_data.csv'), help='Path for saving cluster data')
    parser.add_argument('--path', type=str, default=os.path.join(os.pardir, 'data', 'dummy_data.csv'), help='Path for data')
    
    args = parser.parse_args()
    return args

def get_optimal_params(scaled_data, model, method, method_distance, distance=None):
    if distance:
        distance = (distance.split(" ")[0]).lower()
    method_distance = (method_distance.split(" ")[0]).lower()
    if model == "K-means (default)":
        optimal_k = return_optimal_kmeans(scaled_data, method, method_distance)
        return optimal_k
    elif model == "Hierarchical":
        n, linkage = return_optimal_hierarchical(scaled_data, method, distance, method_distance)
        return n, linkage
    elif model == "GMM":
        n_components, cov_type = return_optimal_gmm(scaled_data, method, method_distance)
        return n_components, cov_type
    elif model == "Spectral Clustering":
        num_clusters, affinity = return_optimal_spectral(scaled_data, method, distance, method_distance)
        return num_clusters, affinity
    elif model == "DBSCAN":
        eps, min_samples = return_optimal_dbscan(scaled_data, method, distance, method_distance)
        return eps, min_samples

def scale_data(scaler, feature_vectors):
    if scaler == "Min-Max (default)":
        X = MinMaxScaler().fit_transform(feature_vectors)
    elif scaler == "Standard":
        X = StandardScaler().fit_transform(feature_vectors)

    return X

def impute_data(data, imputation_method):
    if imputation_method == "Nearest Neighbors (default)":
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        X = imputer.fit_transform(data)
    elif imputation_method == "Zero":
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(data)
    elif imputation_method == "Mean":
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        X = imputer.fit_transform(data)
    elif imputation_method == "Random":
        X = np.copy(data)
        for col in range(data.shape[1]):
            mask = np.isnan(data[:, col])
            if np.any(mask):
                valid_values = data[~mask, col]
                X[mask, col] = np.random.choice(valid_values, size=np.sum(mask))
    
    return X

def cluster_votes(scaled_data, cluster_method, optimization_method, method_distance, distance):
    if distance:
        distance = (distance.split(" ")[0]).lower()
    method_distance = (method_distance.split(" ")[0]).lower()
    if cluster_method == "K-means (default)":
        k = get_optimal_params(scaled_data, cluster_method, optimization_method, method_distance, distance)
        clusters = KMeans(n_clusters=k, random_state=42, n_init="auto")
        clusters.fit(scaled_data)
        return clusters.labels_

    elif cluster_method == "DBSCAN":        
        eps, min_samples = get_optimal_params(scaled_data, cluster_method, optimization_method, method_distance, distance)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=distance)
        cluster_labels = dbscan.fit_predict(scaled_data)                
        
    elif cluster_method == "Hierarchical":
        n, linkage = get_optimal_params(scaled_data, cluster_method, optimization_method, method_distance, distance)
        clustering = AgglomerativeClustering(n_clusters=n, metric=distance, linkage=linkage)
        clustering.fit(scaled_data)
        return clustering.labels_

    elif cluster_method == "GMM":
        best_n_components, best_cov_type = get_optimal_params(scaled_data, cluster_method, optimization_method, method_distance, distance)
        best_gmm = GaussianMixture(n_components=best_n_components, covariance_type=best_cov_type, random_state=42)
        best_gmm.fit(scaled_data)
        cluster_labels = best_gmm.predict(scaled_data)

    elif cluster_method == "Spectral Clustering":
        n_clusters, affinity = get_optimal_params(scaled_data, cluster_method, optimization_method, method_distance, distance)
        spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=42)
        cluster_labels = spectral.fit_predict(scaled_data)

    return cluster_labels

def cluster_percentage(cluster_labels):
    cluster_percentages = []
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    total_voters = len(cluster_labels)
    percentages = (cluster_counts / total_voters).values
    cluster_percentages.append(percentages)
    
    return cluster_percentages

def calculate_vbe(percentages, entropy_fn):
    max_voting_bloc = np.max(percentages)
    
    if entropy_fn == "Min Entropy (default)":
        vbe_value = - np.log2(max_voting_bloc) if max_voting_bloc > 0 else 0
    elif entropy_fn == "Max Entropy":
        vbe_value = np.log2(max_voting_bloc) if max_voting_bloc > 0 else 0
    elif entropy_fn == "Shannon Entropy":
        vbe_value = - max_voting_bloc * np.log2(max_voting_bloc) if max_voting_bloc > 0 else 0

    return vbe_value


def main():
    args = get_args()
    num_clusters = args.num_clusters
    num_clusters_choice = None
    optimal_cluster = {'distance_metric': None, 'optimal_cluster_method': None}

    # Load data
    print("Data path:", args.path)
    load_data_flag = inquirer.text("Do you want to use the above data? (Y/N)", default="Y")
    if load_data_flag != "Y":
        args.path = inquirer.text("Enter the path to the data", default=os.path.join(os.pardir, 'data', 'dummy_data.csv'))
    
    print("Loading data from", args.path)
    feature_vectors, pivot_df = load_data_rpgf(args.path)

    print("Data loaded successfully, now beginning clustering...")
    if not args.model:
        questions1 = [
            inquirer.List(
                "clustering_method",
                message="What clustering method do you want to use?",
                choices=["K-means (default)", "DBSCAN", "Hierarchical", "GMM", "Spectral Clustering"],
            ),
            inquirer.List(
                "imputer",
                message="How would you like to impute null values?",
                choices=["Nearest Neighbors (default)", "Zero", "Mean", "Random"],
            ),
            inquirer.List(
                "scaler",
                message="Would you like to have standard scaling or min-max scaling? Min-Max scaling is recommended for noisy data",
                choices=["Min-Max (default)", "Standard"],
            ),            
            inquirer.List(
                    "entropy_fn",
                    message="What entropy function would you like to use?",
                    choices=["Min Entropy (default)", "Max Entropy", "Shannon Entropy"],
                ),
        ]
        clustering = inquirer.prompt(questions1)

        if clustering['clustering_method'] == "K-means (default)":
            questions2 = [
                inquirer.List(
                    "optimization_method",
                    message="Which method would you like to use for finding the optimal number of clusters?",
                    choices=["Silhouette Score (default)", "Gap Statistic", "Davies-Bouldin Index", "Calinski-Harabasz Index"],
                ),
                inquirer.List(
                    "method_distance",
                    message="Which distance method would you like to use (finding optimal cluster)?",
                    choices=["Euclidean (default)", "Manhattan", "Cosine", "Cityblock", "L1", "L2"],
                ),
            ]
            clustering2 = inquirer.prompt(questions2)
        elif clustering['clustering_method'] == "DBSCAN":
            questions2 = [
                inquirer.List(
                        "distance",
                        message="Which distance method would you like to use (clustering)?",
                        choices=["Euclidean (default)", "Cosine", "Manhattan", "Cityblock", "L1", "L2"], 
                ),
                inquirer.List(
                    "optimization_method",
                    message="Which method would you like to use for finding the optimal number of clusters?",
                    choices=["Silhouette Score (default)"],
                ),
                inquirer.List(
                    "method_distance",
                    message="Which distance method would you like to use (finding optimal cluster)?",
                    choices=["Euclidean (default)", "Manhattan", "Cosine", "Cityblock", "L1", "L2"],
                ),
            ]
            clustering2 = inquirer.prompt(questions2)
        elif clustering['clustering_method'] == "Hierarchical":
            questions2 = [
                inquirer.List(
                        "distance",
                        message="Which distance method would you like to use (clustering)?",
                        choices=["Euclidean (default)", "Cosine", "Manhattan", "Cityblock", "L1", "L2"], 
                ),
                inquirer.List(
                    "optimization_method",
                    message="Which method would you like to use for finding the optimal number of clusters?",
                    choices=["Silhouette Score (default)", "Gap Statistic", "Davies-Bouldin Index", "Calinski-Harabasz Index"],
                ),
                inquirer.List(
                    "method_distance",
                    message="Which distance method would you like to use (finding optimal cluster)?",
                    choices=["Euclidean (default)", "Manhattan", "Cosine", "Cityblock", "L1", "L2"],
                ),
            ]
            clustering2 = inquirer.prompt(questions2) 
        elif clustering['clustering_method'] == "Spectral Clustering":
            questions2 = [
                inquirer.List(
                    "optimization_method",
                    message="Which method would you like to use for finding the optimal number of clusters?",
                    choices=["Silhouette Score (default)", "Davies-Bouldin Index", "Calinski-Harabasz Index"],
                ),
                inquirer.List(
                    "method_distance",
                    message="Which distance method would you like to use (finding optimal cluster)?",
                    choices=["Euclidean (default)", "Manhattan", "Cosine", "Cityblock", "L1", "L2"],
                ),
            ]
            clustering2 = inquirer.prompt(questions2)
        elif clustering['clustering_method'] == "GMM":
            questions2 = [
                inquirer.List(
                    "optimization_method",
                    message="Which method would you like to use for finding the optimal number of clusters?",
                    choices=["Silhouette Score (default)", "BIC", "AIC", "Log-Likelihood"],
                ),
                inquirer.List(
                    "method_distance",
                    message="Which distance method would you like to use (finding optimal cluster)?",
                    choices=["Euclidean (default)", "Manhattan", "Cosine", "Cityblock", "L1", "L2"],
                ),
            ]
            clustering2 = inquirer.prompt(questions2)

        if clustering['clustering_method'] in ['K-means (default)', 'Hierarchical', 'Spectral Clustering']:
            questions3 = [inquirer.Text("num_cluster_input", message="Would you like to run the clustering with the optimal number of clusters? (Y/N)", default="Y")]
            optimal_cluster = inquirer.prompt(questions3)
            
            print("Imputing data...", feature_vectors)
            imputed_feature_vectors = impute_data(feature_vectors, clustering['imputer'])
            print("Imputed data successfully", imputed_feature_vectors)
            scaled_data = scale_data(clustering['scaler'], imputed_feature_vectors)

            if not num_clusters:
                num_clusters, _ = get_optimal_params(scaled_data, clustering['clustering_method'], clustering2['optimization_method'], clustering2['method_distance'], clustering2.get('distance', None))

            print(f"Optimal number of clusters: {num_clusters}")

            if optimal_cluster['num_cluster_input'] not in ["Y", "y", "Yes", "yes"]:
                num_clusters_choice = int(inquirer.text("Enter the number of clusters you would like to use: "))
            else:
                num_clusters_choice = num_clusters
        else:
            optimal_cluster['optimal_cluster_method'] = "Silhouette Score (default)"
        
        save_data_flag = inquirer.text("Save selected parameters and VBE? (Y/N)", default="Y")
        model = clustering['clustering_method']
        method = clustering2['optimization_method']
        method_distance = clustering2['method_distance']
        distance = clustering2.get('distance', None)
        scaler = clustering['scaler']
        entropy_fn = clustering['entropy_fn']        

    else:
        save_data_flag = args.save_data
        save_clusters_flag = args.save_clusters
        save_clusters_path = args.save_cluster_path
        model = args.model
        method = args.optimization_method
        method_distance = args.dist_method
        distance = args.dist_model
        scaler = args.scale
        
    imputed_feature_vectors = impute_data(feature_vectors, clustering['imputer'])
    scaled_data = scale_data(scaler, imputed_feature_vectors)
    cluster_labels = cluster_votes(scaled_data, model, method, method_distance, distance)
    percentages = cluster_percentage(cluster_labels)
    vbe = calculate_vbe(percentages, entropy_fn)

    print("\n Summary of clustering & VBE:",
           "\n --------------------------------------------------------------------",
           "\n Clustering method:                   ", model, 
           "\n Distance method (clustering):        ", distance, 
           "\n Scaler:                              ", scaler, 
           "\n Optimal cluster method:              ", method, 
           "\n Distance method (optimal cluster):   ", method_distance, 
           "\n Optimal clusters:                    ", num_clusters, 
           "\n # of clusters selected:              ", num_clusters_choice,
           "\n Entropy function:                    ", entropy_fn,
           "\n --------------------------------------------------------------------",
           "\n VBE:                                 ", vbe,
           "\n")

    save_to_csv_data = {
        'clustering': model,
        'distance_clustering': distance,
        'scaler': scaler,
        'optimal_cluster_method': method,
        'distance_optimal_cluster': method_distance,
        'optimal_clusters': num_clusters,
        'num_clusters_choice': num_clusters_choice,
        'entropy_function': entropy_fn,
        'vbe': vbe
    }

    if save_data_flag in ["Y", "y", "Yes", "yes"]:
        save_to_csv(save_to_csv_data)
        
    save_clusters_flag = inquirer.text("Save voter cluster data? (Y/N)", default="N")
    if save_clusters_flag in ["Y", "y", "Yes", "yes"]:
        save_clusters_path = inquirer.text("Enter the path to save the cluster data: ", default=os.path.join(os.pardir, 'results', 'cluster_data.csv'))
        save_clusters(pivot_df, cluster_labels, args.path, save_clusters_path)

if __name__ == "__main__":
    main()
