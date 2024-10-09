import csv
import os
import inquirer
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from scipy.spatial.distance import cdist

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import csv
import random
import datetime
import uuid
from typing import List, Dict

def generate_input_data(output_file: str, num_rows: int, total_projects: int):
    headers = [
        "ID", "Has.voted", "Has.published", "Published.at", "Created.at",
        "Updated.at", "Projects.in.ballot", "Votes"
    ]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(headers)

        for _ in range(num_rows):
            row = generate_random_row(total_projects)
            writer.writerow(row)

def generate_random_row(total_projects: int) -> List:
    fake_id = f"ID_{uuid.uuid4().hex[:6]}"
    has_voted = random.choice(["TRUE", "FALSE"])
    has_published = random.choice(["TRUE", "FALSE"])
    
    created_at = datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 30))
    updated_at = created_at + datetime.timedelta(hours=random.randint(1, 48))
    published_at = updated_at if has_published == "TRUE" else ""

    projects_in_ballot = random.randint(10, 100)
    votes = generate_random_votes(total_projects, projects_in_ballot)

    return [
        fake_id, has_voted, has_published,
        published_at.isoformat() + "Z" if published_at else "",
        created_at.isoformat() + "Z",
        updated_at.isoformat() + "Z",
        projects_in_ballot,
        format_votes(votes)
    ]

def generate_random_votes(total_projects: int, num_votes: int) -> List[Dict[str, str]]:
    project_ids = [f"projID_{i}" for i in range(1, total_projects + 1)]
    selected_projects = random.sample(project_ids, num_votes)
    
    votes = []
    for project_id in selected_projects:
        amount = random.randint(100000, 2000000) // 100000 * 100000  # Round to nearest 100,000
        votes.append({"amount": str(amount), "projectId": project_id})
    
    return votes

def format_votes(votes: List[Dict[str, str]]) -> str:
    formatted_votes = [f'{{"amount":"{v["amount"]}","projectId":"{v["projectId"]}"}}' for v in votes]
    return "[" + ",".join(formatted_votes) + "]"

# Usage
output_file = "../data/dummy_data.csv"
num_rows = 150  # Number of rows to generate
total_projects = 120  # Total number of possible projects

generate_input_data(output_file, num_rows, total_projects)
# print(f"CSV file '{output_file}' has been generated with {num_rows} rows.")
    

def save_clusters(data, labels, input_path, filename="../results/cluster_data.csv"):
    orig_data = pd.read_csv(input_path)
    data['voterId'] = orig_data.iloc[:,0].values
    data['cluster'] = labels

    # Move 'cluster' column to the front
    cols = ['voterId', 'cluster'] + [col for col in data.columns if col not in ['voterId', 'cluster']]
    data = data[cols]
    
    # Check if file exists
    file_exists = os.path.isfile(filename)

    # Open the file in append mode ('a') if it exists, otherwise write mode ('w')
    if file_exists:
        overwrite = inquirer.text("Overwrite data? (Y/N)", default="Y")
    
    if not file_exists or overwrite.lower() == 'y':
        data.to_csv(filename, index=False)
        print(f"\nCluster data has been {'overwritten to' if file_exists else 'written to'} {filename}.")

def save_to_csv(data, filename="../results/vbe_results.csv"):
    # Define the column headers
    headers = [
        "Clustering method",
        "Distance method (clustering)",
        "Scaler",
        "Optimal cluster method",
        "Distance method (optimal cluster)",
        "Optimal clusters",
        "# of clusters selected",
        "Entropy function",
        "VBE"
    ]

    # Prepare the row data
    row = [
        data['clustering'],
        data['distance_clustering'],
        data['scaler'],
        data['optimal_cluster_method'],
        data['distance_optimal_cluster'],
        data['optimal_clusters'],
        data['num_clusters_choice'],
        data['entropy_function'],
        data['vbe']
    ]

    # Check if file exists
    file_exists = os.path.isfile(filename)

    # Open the file in append mode ('a') if it exists, otherwise write mode ('w')
    mode = 'a' if file_exists else 'w'
    
    with open(filename, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write headers if the file is being created
        if not file_exists:
            writer.writerow(headers)
        
        # Write the data row
        writer.writerow(row)

    print(f"Data has been {'appended to' if file_exists else 'written to'} {filename}.\n")

def return_optimal_kmeans(scaled_data, method, method_distance):
    k_values = range(2, 11)
    if method == 'Gap Statistic':
        gap_stats = compute_gap_statistic("kmeans", scaled_data, method_distance)
        return k_values[np.argmax(gap_stats)]
    else:
        db_scores = []
        ch_scores = []
        silhouette_scores = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = kmeans.fit_predict(scaled_data)
            silhouette_scores.append(silhouette_score(scaled_data, labels, metric=method_distance))
            db_scores.append(davies_bouldin_score(scaled_data, labels))
            ch_scores.append(calinski_harabasz_score(scaled_data, labels))
    
    # Find optimal k for each method
    if method == 'Davies-Bouldin Index':
        optimal_k = k_values[np.argmin(db_scores)]
    elif method == 'Calinski-Harabasz Index':
        optimal_k = k_values[np.argmax(ch_scores)]
    elif method == 'Silhouette Score (default)':
        optimal_k = k_values[np.argmax(silhouette_scores)]

    return optimal_k

def compute_gap_statistic(model, data, distance, n_references=5):
    """Compute Gap statistic for determining optimal k"""
    k_values = range(2, 11)
    reference = np.random.rand(*data.shape)
    gap_stats = []
    for k in k_values:
        if model == "kmeans":
            # Cluster on real data
            kmeans_real = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans_real.fit(data)
            dispersion_real = sum(np.min(cdist(data, kmeans_real.cluster_centers_, distance), axis=1)) / data.shape[0]

            # Cluster on reference data
            ref_dispersions = []
            for _ in range(n_references):
                kmeans_ref = KMeans(n_clusters=k, n_init=10, random_state=42)
                kmeans_ref.fit(reference)
                dispersion_ref = sum(np.min(cdist(reference, kmeans_ref.cluster_centers_, distance), axis=1)) / reference.shape[0]
                ref_dispersions.append(dispersion_ref)
        elif model == "hierarchical":
            hc_real = AgglomerativeClustering(n_clusters=k)
            hc_real.fit(data)
            dispersion_real = sum(np.min(cdist(data, data[hc_real.labels_ == i].mean(axis=0).reshape(1, -1), 'euclidean') ** 2) 
                                for i in range(k))

            # Cluster on reference data
            ref_dispersions = []
            for _ in range(n_references):
                hc_ref = AgglomerativeClustering(n_clusters=k)
                hc_ref.fit(reference)
                dispersion_ref = sum(np.min(cdist(reference, reference[hc_ref.labels_ == i].mean(axis=0).reshape(1, -1), 'euclidean') ** 2) 
                                    for i in range(k))
                ref_dispersions.append(dispersion_ref)

        gap = np.log(np.mean(ref_dispersions)) - np.log(dispersion_real)
        gap_stats.append(gap)
    
    return gap_stats

def return_optimal_hierarchical(scaled_data, method, distance, method_distance, range=range(2,11)):
    '''
    return optimal parameters for hierarchical clustering, including:
    number of clusters, linkage method, and distance metric
    '''
    linkage_methods = ['ward', 'complete', 'average', 'single']
    if method == 'Gap Statistic':
        # Compute Gap Statistic
        gap_stats = compute_gap_statistic("hierarchical", scaled_data, range)
        # TODO: Change this to return the linkage method as well
        return range[np.argmax(gap_stats)]

    # Compute metrics
    silhouette_scores = {}
    db_scores = {}
    ch_scores = {}
    
    for linkage in linkage_methods:
        for n in range:
            hc = AgglomerativeClustering(n_clusters=n, metric='cosine' if linkage != 'ward' else distance, linkage=linkage)
            labels = hc.fit_predict(scaled_data)
            silhouette_scores[(n, linkage)] = silhouette_score(scaled_data, labels, metric=method_distance)
            db_scores[(n, linkage)] = davies_bouldin_score(scaled_data, labels)
            ch_scores[(n, linkage)] = calinski_harabasz_score(scaled_data, labels)

    # Find optimal k for each method
    if method == 'Davies-Bouldin Index':
        optimal_params = min(db_scores, key=db_scores.get)
    elif method == 'Calinski-Harabasz Index':
        optimal_params = max(ch_scores, key=ch_scores.get)
    elif method == 'Silhouette Score (default)':
        optimal_params = max(silhouette_scores, key=silhouette_scores.get)

    return optimal_params[0], optimal_params[1]

def return_optimal_gmm(scaled_data, method, method_distance, range=range(2,11)):
    covariance_types = ['full', 'tied', 'diag', 'spherical']

    results = {}
    X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)

    for cov_type in covariance_types:
        for n_components in range:
            gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=42)
            gmm.fit(X_train)

            bic = gmm.bic(X_test)
            aic = gmm.aic(X_test)
            log_likelihood = gmm.score(X_test)
            labels = gmm.predict(scaled_data)
            silhouette = silhouette_score(scaled_data, labels, metric=method_distance)
            
            results[(cov_type, n_components)] = (bic, aic, log_likelihood, silhouette)
    
    best_bic = min(results, key=lambda k: results[k][0])
    best_aic = min(results, key=lambda k: results[k][1])
    best_log_likelihood = max(results, key=lambda k: results[k][2])
    best_silhouette = max(results, key=lambda k: results[k][3])

    if method == 'BIC':
        best_cov_type, best_n_components = best_bic
    elif method == 'AIC':
        best_cov_type, best_n_components = best_aic
    elif method == 'Log-Likelihood':
        best_cov_type, best_n_components = best_log_likelihood
    elif method == 'Silhouette Score (default)':
        best_cov_type, best_n_components = best_silhouette

    return best_n_components, best_cov_type
    
def return_optimal_spectral(scaled_data, method, distance, method_distance, range=range(2,11)):
    affinities = ['rbf', 'nearest_neighbors', 'cosine']

    results = {}

    for affinity in affinities:
        for n_clusters in range:
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42, n_neighbors=10)
            labels = spectral.fit_predict(scaled_data)
            
            silhouette = silhouette_score(scaled_data, labels, metric=method_distance)
            calinski = calinski_harabasz_score(scaled_data, labels)
            davies = davies_bouldin_score(scaled_data, labels)
            
            results[(affinity, n_clusters)] = (silhouette, calinski, davies)
            
    best_silhouette = max(results, key=lambda k: results[k][0])
    best_calinski = min(results, key=lambda k: results[k][1])
    best_davies = max(results, key=lambda k: results[k][2])
    
    # Return optimal params for each method
    if method == 'Calinski-Harabasz Index':
        affinity, n_clusters = best_calinski
    elif method == 'Davies-Bouldin Index':
        affinity, n_clusters = best_davies
    elif method == 'Silhouette Score (default)':
        affinity, n_clusters = best_silhouette

    return n_clusters, affinity

def return_optimal_dbscan(scaled_data, method, distance, method_distance, range=range(2,11)):
    best_score = -1
    best_params = None
    
    for eps in np.logspace(-2, 2, 50):
        for min_samples in range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=distance)
            labels = dbscan.fit_predict(scaled_data)                
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters > 1:
                score = silhouette_score(scaled_data, labels, metric=method_distance)
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)

    if method == 'Silhouette Score (default)':
        return best_params
    elif method == 'K-distance':
        # TODO: Implement k-distance method
        return best_params
    
    return eps, min_samples
