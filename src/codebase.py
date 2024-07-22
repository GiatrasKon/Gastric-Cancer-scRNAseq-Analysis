# Importing necessary libraries
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import anndata as ad
import scanpy as sc
from scipy.stats import uniform, randint
from sklearn.model_selection import ParameterSampler
from sklearn.manifold import TSNE, trustworthiness
import umap
from sklearn.metrics import pairwise_distances
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from gprofiler import GProfiler

class GCSingleCellAnalysis:
    def __init__(self, filepath=None):
        """
        Initializes the instance of the class by loading the CSV file specified by the filepath parameter into a pandas DataFrame.
        The DataFrame is then converted to an AnnData object using the `sc.AnnData` constructor.
        The resulting AnnData object is stored as an attribute of the instance.

        Args:
            - filepath (str): The path to the CSV file (default is None).
        """
        self.filepath = filepath # path to the CSV file
        if filepath: # if a filepath is provided
            counts = pd.read_csv(self.filepath, index_col=0) # loading the CSV file into a pandas DataFrame
            self.adata = sc.AnnData(counts.transpose()) # converting the DataFrame to an AnnData object and transposing it to have cells as rows and genes as columns
        else:
            self.adata = None # if no filepath is provided, set the AnnData object to None

    def save_adata(self, filename):
        """
        Saves the AnnData object to a file.

        Args:
            - filename (str): The name of the file to save the AnnData object to.
        """
        self.adata.write(filename) # saving the AnnData object to a h5ad file
        print(f"AnnData object saved to {filename}.") # printing a message indicating that the object was saved
        
    def load_adata(self, filename):
        """
        Loads a saved AnnData object from a file.
        
        Args:
            - filename (str): The name of the file to load the AnnData object from.
        """
        self.adata = sc.read(filename) # loading the AnnData object from a h5ad file
        print(f"AnnData object loaded from {filename}.") # printing a message indicating that the object was loaded

    def preprocess_adata(self):
        """
        Preprocesses the `adata` object by adding columns to it that contain information about mitochondrial,
        ribosomal, and hemoglobin genes. Then calculates quality control metrics for these genes and logs
        the counts.
        """
        self.adata.var["mt"] = self.adata.var_names.str.startswith("MT-") # adding a column to the adata object that contains information about the mitochondrial genes
        self.adata.var["ribo"] = self.adata.var_names.str.startswith(("RPS", "RPL")) # adding a column to the adata object that contains information about the ribosomal genes
        self.adata.var["hb"] = self.adata.var_names.str.contains("^HB[^(P)]") # adding a column to the adata object that contains information about the hemoglobin genes
        sc.pp.calculate_qc_metrics(self.adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True) # calculating QC metrics

    def normalize_adata(self):
        """
        Normalizes the data if it contains raw counts, transforming to TPM and then applying log2 transformation.
        If the data is already normalized, it skips normalization and notifies the user.
        """
        if np.all(self.adata.X.astype(int) == self.adata.X): # checking if the data is raw counts based on the dtype (int)
            if np.any(self.adata.X > 100): # checking if the data contains values greater than 100
                counts_per_million = self.adata.X / self.adata.X.sum(axis=1, keepdims=True) * 1e6 # calculating TPM
                self.adata.X = np.log2(counts_per_million + 1) # applying log2 transformation
                print("Data was normalized to TPM and log2 transformed.")
            else:
                print("Data appears to be truncated TPM or already normalized, skipping normalization.")
        else:
            print("Data is not raw counts, skipping normalization.")

    def filter_adata(self, ribosomal_threshold=50, mitochondrial_threshold=20):
        """
        Filters the data based on specified thresholds for ribosomal and mitochondrial content.

        Args:
            - ribosomal_threshold (int): The threshold for filtering cells based on the percentage of ribosomal reads (defaults=50).
            - mitochondrial_threshold (int): The threshold for filtering cells based on the percentage of mitochondrial reads (default=20).

        Prints:
            - The number of cells after applying gene filtering.
            - The number of cells after filtering high mitochondrial content.
            - The number of cells after filtering high ribosomal content.
        """
        # Gene filtering
        print(f"Number of genes before applying gene filtering: {self.adata.n_vars}")
        self.adata.var['mean_counts'] = self.adata.X.mean(axis=0) # calculating the mean expression (average read count) for each gene across all cells
        self.adata = self.adata[:, self.adata.var['mean_counts'] > 1] # filtering out genes where the average read count is less than or equal to 1
        print(f"Number of genes after applying gene filtering: {self.adata.n_vars}")

        # Mitochondrial content filtering
        print(f"Number of cells before filtering high mitochondrial content: {self.adata.n_obs}")
        self.adata.obs['pct_counts_mt'] = (self.adata[:, self.adata.var['mt']].X.sum(axis=1) / self.adata.X.sum(axis=1)) * 100 # calculating the percentage of mitochondrial reads for each cell
        self.adata = self.adata[self.adata.obs['pct_counts_mt'] < mitochondrial_threshold] # filtering out cells where the percentage of mitochondrial reads is less than the specified threshold
        print(f"Number of cells after filtering high mitochondrial content: {self.adata.n_obs}")

        # Ribosomal content filtering
        print(f"Number of cells before filtering high ribosomal content: {self.adata.n_obs}")
        self.adata.obs['pct_counts_ribo'] = (self.adata[:, self.adata.var['ribo']].X.sum(axis=1) / self.adata.X.sum(axis=1)) * 100 # calculating the percentage of ribosomal reads for each cell
        self.adata = self.adata[self.adata.obs['pct_counts_ribo'] < ribosomal_threshold] # filtering out cells where the percentage of ribosomal reads is less than the specified threshold
        print(f"Number of cells after filtering high ribosomal content: {self.adata.n_obs}")

    def prepare_adata(self):
        """
        Preprocesses the AnnData object to extract patient ID and tissue type,
        combines them into a 'cell_type' column, and renames the index.
        """
        self.adata.obs_names = self.adata.obs_names.str.strip() # stripping the whitespace from the cell IDs
        self.adata.obs['patient'] = self.adata.obs_names.str.extract('(GC\d+)', expand=False) # extracting the patient ID from the cell IDs
        self.adata.obs['tissue'] = self.adata.obs_names.str.extract('-(TT|LN)', expand=False) # extracting the tissue type from the cell IDs
        self.adata.obs['cell_type'] = self.adata.obs['patient'] + "-" + self.adata.obs['tissue'] # combining the patient ID and tissue type to form the 'cell_type' column
        self.adata.obs.index.name = 'cell_id' # setting the index name of the DataFrame to 'cell_id'
        
    def perform_pca(self, n_comps=50):
        """
        Performs Principal Component Analysis (PCA) on the preprocessed dataset and stores the results.
        
        Args:
            - n_comps (int): The number of principal components to retain (default=50).
        """
        max_components = min(self.adata.n_obs, self.adata.n_vars) - 1 # determining the maximum number of components to use
        if n_comps > max_components: # checking if the requested number of components is greater than the maximum allowed
            print(f"Requested n_comps ({n_comps}) is greater than the maximum allowed ({max_components}). Setting n_comps to {max_components}.")
            n_comps = max_components # setting n_comps to the maximum allowed value
        
        sc.tl.pca(self.adata, n_comps=n_comps, svd_solver='arpack') # performing PCA with the appropriate number of components
        self.adata.uns['pca_variance_ratio'] = self.adata.uns['pca']['variance_ratio'] # storing the PCA variance ratios in the adata object as an uns attribute
        print(f"PCA completed with {n_comps} components.")

    def determine_optimal_pcs(self, variance_threshold=0.90):
        """
        Determines the optimal number of principal components to capture the specified percent of explained variance.
        
        Args:
            - variance_threshold (float): The desired percentage of explained variance (default=0.90).
            
        Returns:
            - optimal_pcs (int): The optimal number of principal components to retain.
        """
        if 'pca_variance_ratio' not in self.adata.uns: # checking if PCA has been performed
            print("PCA variance ratios are not available. Run perform_pca() first.") # if not, raise an error
            return

        variance_ratios = self.adata.uns['pca_variance_ratio']  # extracting the variance ratios from the adata object
        cumulative_variance = np.cumsum(variance_ratios)  # calculating the cumulative variance

        # Checking if total cumulative variance is less than expected
        print(f"Total Cumulative Variance with chosen number of components: {cumulative_variance[-1]*100:.2f}%")

        # Determining the number of components to explain various levels of total variance
        thresholds = [0.7, 0.8, 0.9]  # 70%, 80%, and 90% thresholds
        components_required = {} # dictionary to store the number of components required for each threshold
        # Looping through the thresholds
        for threshold in thresholds: 
            indices = np.where(cumulative_variance >= threshold)[0] # finding the indices where the cumulative variance is greater than or equal to the threshold
            if indices.size > 0: # checking if any indices are found
                components = indices[0] + 1 # adding 1 to the first index to get the number of components
                components_required[threshold] = components # storing the number of components for the current threshold
                print(f"To explain {threshold*100:.0f}% of the variance, {components} components are required.") # printing the number of components required for the current threshold
            else: # if no indices are found
                print(f"Not enough variance captured to explain {threshold*100:.0f}% with the given number of components.")

        # Determining the optimal number of PCs to capture the specified variance threshold
        optimal_pcs = np.argmax(cumulative_variance >= variance_threshold) + 1 if cumulative_variance[-1] >= variance_threshold else len(cumulative_variance) # finding the index of the first element in cumulative_variance that is greater than or equal to variance_threshold
        print(f"Number of PCs to explain the chosen {variance_threshold*100:.0f}% variance: {optimal_pcs}") # printing the optimal number of PCs
        
        # Plotting the cumulative variance
        plt.figure(figsize=(10, 6)) # setting the figure size
        plt.plot(cumulative_variance, marker='o', linestyle='-', color='b') # plotting the cumulative variance
        plt.title('Cumulative Variance Explained by PCA Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Explained')

        # Annotating the thresholds on the plot
        for threshold, components in components_required.items(): # looping through the components required for each threshold
            plt.axhline(y=threshold, color='r', linestyle='--') # plotting the vertical line at the threshold
            plt.annotate(f'{components} components ({threshold*100:.0f}%)', xy=(components, threshold), xytext=(components+5, threshold),
                        arrowprops=dict(facecolor='black', arrowstyle='->'), verticalalignment='bottom') # annotating the number of components required for the current threshold

        plt.grid(True, which='both', linestyle='--', linewidth=0.5) # adding grid lines
        plt.gca().set_axisbelow(True) # ensuring grid lines are behind other plot elements
        plt.grid(True)
        plt.show()

        return optimal_pcs  # returning the optimal number of PCs

    def prepare_pca_reduced_adata(self, variance_threshold=0.90):
        """
        Prepares the reduced dataset by keeping the optimal number of PCs.
        
        Args:
            - variance_threshold (float): The desired percentage of explained variance (default=0.90).
        """
        optimal_pcs = self.determine_optimal_pcs(variance_threshold) # determining the optimal number of PCs
        # Performing PCA again on the original preprocessed data with the optimal number of components
        original_adata = self.adata.copy()
        sc.tl.pca(original_adata, n_comps=optimal_pcs, svd_solver='arpack') # performing PCA on the preprocessed dataset
        self.adata.obsm['X_pca'] = original_adata.obsm['X_pca'] # storing the PCA results in the adata object
        print(f"PCA completed with {optimal_pcs} components.") 
        print("Shape of PCA results:", self.adata.obsm['X_pca'].shape) # printing the shape of the PCA results
        
    def plot_pca(self):
        """
        Plots pairwise PCA scatter plots and histograms for the first 3 PCs.
        """
        if 'X_pca' not in self.adata.obsm: # checking if PCA has been performed
            print("PCA has not been performed yet. Run perform_pca() first.")
            return

        fig, axes = plt.subplots(3, 3, figsize=(15, 15)) # creating a 3x3 grid of subplots

        # Plotting histograms and scatter plots
        for i in range(3):
            for j in range(3):
                ax = axes[i, j] # getting the current subplot
                if i == j: # if the subplot is on the diagonal
                    sns.histplot(self.adata.obsm['X_pca'][:, i], kde=True, ax=ax) # plotting the histogram for the current PC
                    ax.set_title(f'PC{i+1} Distribution') # setting the title of the subplot to indicate the PC number
                    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)  # adding horizontal grid lines
                    ax.set_axisbelow(True) # ensuring grid lines are behind other plot elements
                else: # if the subplot is not on the diagonal
                    scatter = ax.scatter(self.adata.obsm['X_pca'][:, j], self.adata.obsm['X_pca'][:, i],
                                        c=self.adata.obs['cell_type'].astype('category').cat.codes, # coloring by cell_type
                                        alpha=0.7, s=30, cmap='Spectral') # using a colormap for better visualization
                    ax.set_xlabel(f'PC{j+1}') # setting the x-axis label to indicate the PC number
                    ax.set_ylabel(f'PC{i+1}') # setting the y-axis label to indicate the PC number
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5) # adding grid lines
                    ax.set_axisbelow(True) # ensuring grid lines are behind other plot elements

        # Adding legends
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=cell_type,
                                markerfacecolor=plt.cm.Spectral(i / (len(self.adata.obs['cell_type'].unique()) - 1)),  # using a colormap for better visualization
                                markersize=10)
                        for i, cell_type in enumerate(self.adata.obs['cell_type'].unique())]  # creating a legend for each cell_type
        axes[0, 0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.05, 1))  # adding the legend to the top right corner of the first subplot

        plt.tight_layout()
        plt.show()

    def optimize_tsne(self, n_iter=30, random_state=42):
        """
        Optimizes t-SNE hyperparameters based on the trustworthiness metric.

        Args:
            - n_iter (int): Number of parameter settings that are sampled.
            - random_state (int): The seed used by the random number generator.
        
        Returns:
            - best_params (dict): The best t-SNE parameters.
        """
        param_grid = {
            'perplexity': randint(5, 51),
            'n_iter': randint(250, 5001),
            'learning_rate': uniform(10, 1000),
            'random_state': randint(1, 101)
        } # defining the parameter grid for t-SNE
        
        best_trustworthiness = -np.inf # initializing the best trustworthiness score
        best_params = None # initializing the best parameters
        
        param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state)) # sampling the parameter grid
        
        # Looping through the parameter list
        for params in param_list:
            tsne = TSNE(n_components=2, **params) # initializing the t-SNE object with the current parameters
            embedding = tsne.fit_transform(self.adata.X) # performing t-SNE on the preprocessed dataset
            score = trustworthiness(self.adata.X, embedding) # calculating the trustworthiness score
            
            if score > best_trustworthiness: # if the current trustworthiness score is better than the best trustworthiness score
                best_trustworthiness = score # updating the best trustworthiness score
                best_params = params # updating the best parameters
        
        print(f"Best t-SNE params: {best_params} with trustworthiness score: {best_trustworthiness}") # printing the best t-SNE parameters and trustworthiness score
        self.tsne_params = best_params # storing the best t-SNE parameters in the adata object
        return best_params # returning the best t-SNE parameters

    def perform_tsne(self, optimize=False, n_iter=30, random_state=42, **kwargs):
        """
        Performs t-SNE on the dataset and stores the results.

        Args:
            - optimize (bool, optional): Whether to optimize hyperparameters.
            - n_iter (int): Number of parameter settings that are sampled for optimization.
            - random_state (int): The seed used by the random number generator.
            - **kwargs: Additional arguments for t-SNE.
        """
        if optimize: # if optimization is enabled
            params = self.optimize_tsne(n_iter=n_iter, random_state=random_state) # optimizing t-SNE hyperparameters
        else: # if optimization is disabled
            params = { 
                'perplexity': kwargs.get('perplexity', 30), 
                'n_iter': kwargs.get('n_iter', 5000),
                'learning_rate': kwargs.get('learning_rate', 200),
                'random_state': random_state
            } # default values for t-SNE hyperparameters
        
        tsne = TSNE(n_components=2, **params) # initializing the t-SNE object with the specified parameters
        self.adata.obsm['X_tsne'] = tsne.fit_transform(self.adata.X) # performing t-SNE on the preprocessed dataset and storing the t-SNE coordinates in the adata object
        print("t-SNE completed.")
        
    def plot_tsne(self):
        """
        Plots the t-SNE results.
        """
        if 'X_tsne' not in self.adata.obsm: # checking if t-SNE has been performed
            print("t-SNE has not been performed yet. Run perform_tsne() first.")
            return
        
        # Plotting the t-SNE results
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(self.adata.obsm['X_tsne'][:, 0], self.adata.obsm['X_tsne'][:, 1],
                            c=self.adata.obs['cell_type'].astype('category').cat.codes, 
                            alpha=0.7, s=30, cmap='Spectral')  # using Spectral colormap for better visualization
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE Plot')

        # Adding legends
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=cell_type,
                                markerfacecolor=plt.cm.Spectral(i / (len(self.adata.obs['cell_type'].unique()) - 1)), 
                                markersize=10) # using Spectral colormap for better visualization
                        for i, cell_type in enumerate(self.adata.obs['cell_type'].unique())] 
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.05, 1)) # adding legends to the top right corner

        plt.grid(True, which='both', linestyle='--', linewidth=0.5) # adding grid lines
        plt.gca().set_axisbelow(True) # ensuring grid lines are behind other plot elements
        plt.tight_layout()
        plt.show()

    def optimize_umap(self, n_iter=30, random_state=42):
        """
        Optimizes UMAP hyperparameters based on the trustworthiness metric.

        Args:
            - n_iter (int): Number of parameter settings that are sampled.
            - random_state (int): The seed used by the random number generator.
            
        Returns:
            - best_params (dict): The best UMAP parameters.
        """
        param_grid = { 
            'n_neighbors': randint(5, 51),
            'min_dist': uniform(0.001, 0.5),
            'metric': ['euclidean', 'manhattan', 'mahalanobis'],
            'random_state': randint(1, 101)
        } # defining the parameter grid for UMAP
        
        best_trustworthiness = -np.inf # initializing the best trustworthiness score
        best_params = None # initializing the best parameters
        
        param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state)) # sampling the parameter grid
        
        # Looping through the parameter list
        for params in param_list:
            reducer = umap.UMAP(n_components=2, **params) # initializing the UMAP object with the current parameters
            embedding = reducer.fit_transform(self.adata.X) # performing UMAP on the preprocessed dataset
            score = trustworthiness(self.adata.X, embedding) # calculating the trustworthiness score
            
            if score > best_trustworthiness: # if the current trustworthiness score is better than the best trustworthiness score
                best_trustworthiness = score # updating the best trustworthiness score
                best_params = params # updating the best parameters
        
        print(f"Best UMAP params: {best_params} with trustworthiness score: {best_trustworthiness}") # printing the best UMAP parameters and trustworthiness score
        self.umap_params = best_params # storing the best UMAP parameters in the adata object
        return best_params # returning the best UMAP parameters

    def perform_umap(self, optimize=False, n_iter=30, random_state=42, **kwargs):
        """
        Performs UMAP on the dataset and stores the results.

        Args:
            - optimize (bool, optional): Whether to optimize hyperparameters.
            - n_iter (int): Number of parameter settings that are sampled for optimization.
            - random_state (int): The seed used by the random number generator.
            - **kwargs: Additional arguments for UMAP.
        """
        if optimize: # if optimization is enabled
            params = self.optimize_umap(n_iter=n_iter, random_state=random_state) # optimizing UMAP hyperparameters
        else: # if optimization is disabled
            params = {
                'n_neighbors': kwargs.get('n_neighbors', 15),
                'min_dist': kwargs.get('min_dist', 0.1),
                'metric': kwargs.get('metric', 'euclidean'),
                'random_state': random_state
            } # default values for UMAP hyperparameters
        
        reducer = umap.UMAP(n_components=2, **params) # initializing the UMAP object
        self.adata.obsm['X_umap'] = reducer.fit_transform(self.adata.X) # performing UMAP on the preprocessed dataset and storing the UMAP coordinates in the adata object
        print("UMAP completed.")

    def plot_umap(self):
        """
        Plots the UMAP results.
        """
        if 'X_umap' not in self.adata.obsm: # checking if UMAP has been performed
            print("UMAP has not been performed yet. Run perform_umap() first.")
            return
        
        # Plotting the UMAP results
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(self.adata.obsm['X_umap'][:, 0], self.adata.obsm['X_umap'][:, 1],
                              c=self.adata.obs['cell_type'].astype('category').cat.codes, 
                              alpha=0.7, s=30, cmap='Spectral')  # using Spectral colormap for better visualization
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('UMAP Plot')

        # Adding legends
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=cell_type,
                                  markerfacecolor=plt.cm.Spectral(i / (len(self.adata.obs['cell_type'].unique()) - 1)), 
                                  markersize=10) # using Spectral colormap for better visualization
                           for i, cell_type in enumerate(self.adata.obs['cell_type'].unique())] 
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.05, 1)) # adding legends to the top right corner

        plt.grid(True, which='both', linestyle='--', linewidth=0.5) # adding grid lines
        plt.gca().set_axisbelow(True) # ensuring grid lines are behind other plot elements
        plt.tight_layout()
        plt.show()

    def apply_clustering(self, data, method='gmm', n_clusters=4, embedding='X_pca', optimize=False):
        """
        Applies clustering to the given data using the specified method.

        Args:
            - data (array-like): The input data to be clustered.
            - method (str, optional): The clustering method to be used. Defaults to 'gmm'.
            - n_clusters (int, optional): The number of clusters to be generated. Defaults to 4.
            - embedding (str, optional): The embedding to be used. Defaults to 'X_pca'.
            - optimize (bool, optional): Whether to optimize the number of clusters. Defaults to False.

        Returns:
            - labels (array-like): The cluster labels assigned to each data point.

        Raises:
            ValueError: If the specified method is unknown.
        """
        best_silhouette = -1 # initializing the best Silhouette score to -1
        best_labels = None # initializing the best cluster labels to None
        best_n_clusters = n_clusters # initializing the best number of clusters to n_clusters

        if optimize and method in ['gmm', 'average_link', 'ward', 'spectral']:  # if optimization is enabled and method is GMM, Average Linkage, Ward, or Spectral
            for k in range(2, 11):  # looping through different numbers of clusters
                if method == 'gmm':  # if Gaussian Mixture Model is selected
                    model = GaussianMixture(n_components=k, random_state=0)  # initializing the Gaussian Mixture Model
                    labels = model.fit_predict(data)  # performing GMM on the reduced dataset
                elif method == 'average_link':  # if Average Linkage Clustering is selected
                    model = AgglomerativeClustering(n_clusters=k, linkage='average')  # initializing the Average Linkage Clustering
                    labels = model.fit_predict(data)  # performing Average Linkage Clustering on the reduced dataset
                elif method == 'ward':  # if Ward Linkage Clustering is selected
                    model = AgglomerativeClustering(n_clusters=k, linkage='ward')  # initializing the Ward Linkage Clustering
                    labels = model.fit_predict(data)  # performing Ward Linkage Clustering on the reduced dataset
                elif method == 'spectral':  # if Spectral Clustering is selected
                    model = SpectralClustering(n_clusters=k, random_state=0, affinity='nearest_neighbors')  # initializing the Spectral Clustering
                    model.fit(data)  # performing Spectral Clustering on the reduced dataset
                    labels = model.labels_  # getting the cluster labels
                silhouette_avg = silhouette_score(data, labels) if len(set(labels)) > 1 else -1  # calculating the Silhouette score for the current number of clusters
                if silhouette_avg > best_silhouette:  # if the current Silhouette score is better than the best Silhouette score
                    best_silhouette = silhouette_avg
                    best_labels = labels
                    best_n_clusters = k
            print(f"Best number of clusters for {method}: {best_n_clusters} with silhouette score: {best_silhouette}")
            return best_labels
        else:
            if method == 'gmm':  # if Gaussian Mixture Model is selected
                model = GaussianMixture(n_components=n_clusters, random_state=0)  # initializing the Gaussian Mixture Model
                labels = model.fit_predict(data)  # performing GMM on the reduced dataset
            elif method == 'average_link':  # if Average Linkage Clustering is selected
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')  # initializing the Average Linkage Clustering
                labels = model.fit_predict(data)  # performing Average Linkage Clustering on the reduced dataset
            elif method == 'ward':  # if Ward Linkage Clustering is selected
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')  # initializing the Ward Linkage Clustering
                labels = model.fit_predict(data)  # performing Ward Linkage Clustering on the reduced dataset
            elif method == 'spectral':  # if Spectral Clustering is selected
                model = SpectralClustering(n_clusters=n_clusters, random_state=0, affinity='nearest_neighbors')  # initializing the Spectral Clustering
                model.fit(data)  # performing Spectral Clustering on the reduced dataset
                labels = model.labels_  # getting the cluster labels
            elif method == 'louvain':  # if Louvain Clustering is selected
                sc.pp.neighbors(self.adata, use_rep=embedding)  # performing Nearest Neighbors on the reduced dataset to prepare for Louvain Clustering
                sc.tl.louvain(self.adata, resolution=1.0)  # performing Louvain Clustering on the reduced dataset
                labels = self.adata.obs['louvain'].to_numpy()  # getting the cluster labels
            elif method == 'leiden':  # if Leiden Clustering is selected
                sc.pp.neighbors(self.adata, use_rep=embedding)  # performing Nearest Neighbors on the reduced dataset to prepare for Leiden Clustering
                sc.tl.leiden(self.adata, resolution=1.0)  # performing Leiden Clustering on the reduced dataset
                labels = self.adata.obs['leiden'].to_numpy()  # getting the cluster labels
            else:  # if unknown method is selected
                raise ValueError("Unknown method!")
            return labels  # returning the cluster labels

    def evaluate_clustering(self, labels, data):
        """
        Evaluates the clustering of the given data using the specified labels.

        Args:
            - labels (array-like): The cluster labels assigned to each data point.
            - data (array-like): The input data to be clustered.

        Returns:
            - dict: A dictionary containing the evaluation metrics for the clustering. The keys are the names of the metrics and the values are the corresponding scores. The metrics included are 'Silhouette', 'Calinski-Harabasz', and 'Davies-Bouldin'. If the number of unique labels is less than or equal to 1, the Silhouette score is set to -1.
        """
        sil = silhouette_score(data, labels) if len(set(labels)) > 1 else -1 # calculating the Silhouette score
        ch = calinski_harabasz_score(data, labels) # calculating the Calinski-Harabasz index
        db = davies_bouldin_score(data, labels) # calculating the Davies-Bouldin score
        return {'Silhouette': sil, 'Calinski-Harabasz': ch, 'Davies-Bouldin': db} # returning the evaluation metrics

    def cluster_and_evaluate(self, methods, embeddings=['X_pca'], optimize=False, n_clusters=4):
        """
        Clusters and evaluates data using the specified methods and embeddings.

        Args:
            - methods (list): A list of clustering methods to be applied.
            - embeddings (list, optional): A list of embeddings to be used for clustering. Defaults to 'X_pca'.
            - optimize (bool, optional): Whether to optimize the number of clusters. Defaults to False.

        Returns:
            - results (dict): A dictionary containing the results for each method and embedding. The keys are the clustering methods and the values are dictionaries with the evaluation metrics for each embedding.
        """
        results = {} # dictionary to store the results
        for method in methods: # looping through the specified methods
            method_results = {} # dictionary to store the results for the current method
            print(f"Clustering with {method}...") # printing the current method
            for emb in embeddings: # looping through the specified embeddings
                print(f"Using {emb} embeddings...") # printing the current embedding
                labels = self.apply_clustering(self.adata.obsm[emb], method=method, embedding=emb, optimize=optimize, n_clusters=n_clusters) # applying clustering to the current embedding
                self.adata.obs[f'{method}_{emb}'] = labels.astype(str) # adding the cluster labels to the adata object
                eval_metrics = self.evaluate_clustering(labels, self.adata.obsm[emb]) # evaluating the clustering results for the current embedding
                method_results[emb] = eval_metrics # adding the evaluation metrics to the dictionary

                # Plotting each embedding with its cluster labels
                fig, ax = plt.subplots(figsize=(8, 8))
                sc.pl.embedding(self.adata, basis=emb, color=f'{method}_{emb}', title=f'{method.upper()} on {emb.upper()}', ax=ax, show=False, size=300)
                
                ax.set_xlabel('Component 1') # setting the x-axis label
                ax.set_ylabel('Component 2') # setting the y-axis label
                # Adding top and right spines
                ax.spines['top'].set_visible(True) 
                ax.spines['right'].set_visible(True) 
                ax.tick_params(axis='both', which='major', labelsize=10) # setting the tick parameters
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # setting the x-axis ticks to integers
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True)) # setting the y-axis ticks to integers
                ax.grid(True, linestyle='--', linewidth=0.5) # adding grid lines
                ax.set_axisbelow(True) # ensuring grid lines are behind other plot elements
                
                plt.tight_layout()
                plt.show()

            results[method] = method_results # adding the clustering results to the results dictionary
        return results # returning the results

    def create_results_dataframe(self, results):
        """
        Creates a DataFrame from the given `results` dictionary.

        Args:
            - results (dict): A dictionary containing the clustering results.
                The keys are the reduction methods, and the values are dictionaries.
                These dictionaries contain the clustering methods as keys and a nested dictionary
                as values. The nested dictionary contains the embeddings as keys and a dictionary
                of evaluation metrics as values.

        Returns:
            - results_df (pandas.DataFrame): A DataFrame containing the flattened results.
                The DataFrame has the following columns:
                - 'Embedding': The embedding used for clustering.
                - 'Clustering Algorithm': The clustering method used.
                - 'Silhouette Score': The silhouette score for the clustering.
                - 'Calinski-Harabasz Index': The Calinski-Harabasz index for the clustering.
                - 'Davies-Bouldin Score': The Davies-Bouldin score for the clustering.

                The scores are rounded to 3 decimal places for better presentation.
        """
        # Flattening the results dictionary into a list of dictionaries for easier conversion to DataFrame
        data = [] # list to store the flattened results
        for reduction_method, method_results in results.items(): # looping through the reduction methods
            for method, embeddings in method_results.items(): # looping through the clustering methods
                for emb, metrics in embeddings.items(): # looping through the embeddings
                    entry = {
                        'Embedding': emb,
                        'Clustering Algorithm': method,
                        'Silhouette Score': metrics['Silhouette'],
                        'Calinski-Harabasz Index': metrics['Calinski-Harabasz'],
                        'Davies-Bouldin Score': metrics['Davies-Bouldin']
                    } # creating a dictionary with the evaluation metrics
                    data.append(entry) # adding the dictionary to the list

        results_df = pd.DataFrame(data) # converting the list to a DataFrame
        # Rounding the scores for better presentation
        results_df['Silhouette Score'] = results_df['Silhouette Score'].round(3)
        results_df['Calinski-Harabasz Index'] = results_df['Calinski-Harabasz Index'].round(3)
        results_df['Davies-Bouldin Score'] = results_df['Davies-Bouldin Score'].round(3)
        
        return results_df # returning the DataFrame

    def plot_clustering_evaluation(self, results_df):
        """
        Generates a bar plot to evaluate the clustering results.

        Args:
            - results_df (pandas.DataFrame): A DataFrame containing the results of clustering evaluation.
        """
        palette = sns.color_palette("viridis", n_colors=len(results_df['Embedding'].unique())) # defining a color palette based on the number of unique embeddings
        metrics = ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Score'] # list of metrics to plot
        
        fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(8, 9)) # creating a grid of subplots
        
        for i, metric in enumerate(metrics): # looping through the metrics
            sns.barplot(data=results_df, x='Clustering Algorithm', y=metric, hue='Embedding', ax=axes[i],
                        palette=palette, width=0.6) # creating a bar plot for the current metric
            axes[i].set_title(f'{metric} Scores by Clustering Algorithm and Embedding Combinations') # setting the title of the subplot
            axes[i].set_ylabel('Score') # setting the y-axis label
            axes[i].set_xlabel('') # setting the x-axis label to an empty string

            # Only displaying the legend on the last plot to avoid redundancy
            if i < len(metrics) - 1: # if the current plot is not the last one
                axes[i].get_legend().remove() # removing the legend
            else: # if the current plot is the last one
                axes[i].legend(title='Embedding') # adding a title to the legend

            axes[i].grid(True, axis='y', which='both', linestyle='--', linewidth=0.5) # adding grid lines
            axes[i].set_axisbelow(True) # ensuring grid lines are behind other plot elements

        axes[-1].set_xlabel('Clustering Method') # setting the x-axis label to the last plot as a common label

        plt.tight_layout(pad=2.0)  # Add padding to ensure everything fits without overlap
        plt.show()
        
    def analyze_and_plot_markers(self, group_key, method='wilcoxon', n_genes=10, filename='marker_genes.csv', figsize=(10, 7)):
        """
        Analyzes marker genes from the AnnData object and plots a heatmap of ranked genes.

        Args:
            - group_key (str): The key in adata.obs that corresponds to the cluster.
            - method (str): The statistical method for ranking genes (e.g. 't-test', 'logreg'). Default is 'wilcoxon'.
            - n_genes (int): Number of top marker genes for each cluster to plot.
            - filename (str): Path to save the marker genes CSV.
            - figsize (tuple): Dimensions of the heatmap figure.
        """
        # Normalizing data if not already log-transformed
        if 'log1p' not in self.adata.uns_keys(): # checking if the 'log1p' uns key exists
            sc.pp.log1p(self.adata) # log-transforming the data

        # Ranking marker genes 
        sc.tl.rank_genes_groups(self.adata, group_key, method=method)
        
        # Extracting results into a DataFrame
        result = self.adata.uns['rank_genes_groups'] # extracting the results from the 'rank_genes_groups' uns key
        groups = result['names'].dtype.names # getting the names of the groups
        df = pd.DataFrame(
            {group + '_' + key[:15]: result[key][group]
            for group in groups for key in ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']}
        ) # creating a DataFrame from the results
        
        # Saving the DataFrame to CSV
        df.to_csv(filename) 
        print(f"Marker genes saved to {filename}") # printing a message indicating that the marker genes have been saved

        # Printing the top genes for each cluster
        print("Top marker genes for each cluster:")
        for group in groups:
            top_genes = result['names'][group][:n_genes]
            print(f"\nCluster {group}:")
            for i, gene in enumerate(top_genes, 1):
                print(f"{i}. {gene}")

        # Plotting the heatmap
        sc.pl.rank_genes_groups_heatmap(
            self.adata, # passing the AnnData object
            swap_axes=True, # swapping the x and y axes
            cmap='viridis', # using a viridis colormap
            vmin=0, # setting the minimum value to 0
            vmax=3, # setting the maximum value to 3
            n_genes=n_genes, # setting the number of genes to plot
            figsize=figsize, # setting the figure size
            dendrogram=False, # disabling the dendrogram
            show_gene_labels=True, # displaying the gene labels
            show=False # disabling the automatic display
        )
        plt.suptitle(f'Heatmap of the Top {n_genes} Highly Expressed Marker Genes in Each Cluster', fontsize=16) # setting the title of the heatmap
        plt.show() # displaying the heatmap
        
    def fetch_go_annotations(self, group_key, method='wilcoxon', n_genes=10, organism='hsapiens'):
        """
        Fetches GO annotations for the top marker genes using g:Profiler.

        Args:
            - group_key (str): The key in adata.obs that corresponds to the grouping.
            - method (str): The statistical method for ranking genes. Default is 'wilcoxon'.
            - n_genes (int): Number of top genes to fetch annotations for.
            - organism (str): Organism name for g:Profiler (default is 'hsapiens' for human).

        Returns:
            - go_annotations (dict): A dictionary with group names as keys and DataFrames of GO annotations as values.
        """
        # Ranking marker genes
        sc.tl.rank_genes_groups(self.adata, group_key, method=method)
        
        # Extracting results into a DataFrame
        result = self.adata.uns['rank_genes_groups'] # extracting the results from the 'rank_genes_groups' uns key
        groups = result['names'].dtype.names # getting the names of the groups

        top_genes = {} # dictionary to store the top genes for each group
        for group in groups: # looping through the clusters
            top_genes[group] = result['names'][group][:n_genes].tolist() # extracting the top genes for the current cluster

        # Initializing g:Profiler
        gp = GProfiler(return_dataframe=True)

        # Fetching GO annotations
        go_annotations = {} # dictionary to store the GO annotations
        for group, genes in top_genes.items(): # looping through the top genes for each cluster
            res = gp.profile(organism=organism, query=genes) # fetching the GO annotations for the current cluster
            if res.shape[0] < n_genes: # checking if the number of GO annotations is less than n_genes
                print(f"Warning: Only {res.shape[0]} GO annotations found for group {group}.") # printing a warning message
            go_annotations[group] = res # adding the GO annotations to the dictionary

        return go_annotations # returning the GO annotations dictionary

    def print_go_annotations(self, go_annotations):
        """
        Prints GO annotations.

        Args:
            - go_annotations (dict): A dictionary with group names as keys and DataFrames of GO annotations as values.
        """
        for group, df in go_annotations.items(): # looping through the GO annotations for each cluster
            print(f"\nGO annotations for group {group}:") # printing the current cluster
            display(df.head(10)) # displaying the top 10 GO annotations for the current cluster
            
            
# USAGE EXAMPLE

# PREPROCESSING
# analysis = GCSingleCellAnalysis('counts_matrix.csv') # creating an instance of the GCSingleCellAnalysis class
# analysis.preprocess_adata() # preprocessing the data and calculating QC metrics
# analysis.normalize_adata() # normalizing the data (TPM and log2 transformation)
# analysis.filter_adata() # filtering the data based on the QC metrics
# analysis.prepare_adata() # preparing the AnnData object for further analysis

# DIMENSIONALITY REDUCTION
# analysis.perform_pca() # performing PCA on the preprocessed dataset
# analysis.prepare_pca_reduced_adata() # performing PCA on the preprocessed dataset with the computed optimal number of components
# analysis.plot_pca() # plotting the PCA results in a pairplot
# analysis.perform_tsne() # performing t-SNE on the preprocessed dataset
# analysis.plot_tsne() # plotting the t-SNE results
# analysis.perform_umap() # performing UMAP on the preprocessed dataset
# analysis.plot_umap() # plotting the UMAP results

# CLUSTERING
# methods = ['gmm', 'average_link', 'ward', 'spectral', 'louvain', 'leiden'] # defining the clustering methods
# results_pca = analysis.cluster_and_evaluate(methods, embeddings=['X_pca']) # clustering and evaluating the PCA reduced data
# results_tsne = analysis.cluster_and_evaluate(methods, embeddings=['X_tsne']) # clustering and evaluating the t-SNE reduced data
# results_umap = analysis.cluster_and_evaluate(methods, embeddings=['X_umap']) # clustering and evaluating the UMAP reduced data
# combined_results = {'PCA': results_pca, 't-SNE': results_tsne, 'UMAP': results_umap} # combining all clustering results
# results_df = analysis.create_results_dataframe(combined_results) # creating a DataFrame for results
# analysis.plot_clustering_evaluation(results_df) # plotting the clustering evaluation results

# POST-CLUSTERING ANALYSIS
# analysis.analyze_and_plot_markers(group_key='average_link_X_umap', n_genes=10) # identifying the top 10 marker genes for each cluster based on the best clustering results
# go_annotations = analysis.fetch_go_annotations(group_key='average_link_X_umap', n_genes=10) # fetching GO annotations for the top 10 marker genes for each cluster based on the best clustering results
# analysis.print_go_annotations(go_annotations) # printing the GO annotations