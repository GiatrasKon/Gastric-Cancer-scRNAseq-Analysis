
# Gastric Cancer Single-Cell RNA-seq Analysis

This repository contains the code and data for the recreation and enrichment of the gastric (GC) cancer single-cell RNA-seq (scRNA-seq) data analysis pipeline described in the ["Comprehensive analysis of metastatic gastric cancer tumour cells using single‑cell RNA‑seq" by Wang B. et. al](https://www.nature.com/articles/s41598-020-80881-2), using the raw counts matrix that is available in [GEO (GSE158631)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158631). Our robust and comprehensive pipeline surpasses previous approaches by incorporating multiple dimensionality reduction techniques, various clustering methods, marker gene identification and GO functional annotation.

This analysis was performed as part of the final project for the "Machine Learning in Computational Biology" graduate course of the MSc Data Science & Information Technologies Master's programme (Bioinformatics - Biomedical Data Science Specialization) of the Department of Informatics and Telecommunications department of the National and Kapodistrian University of Athens (NKUA), under the supervision of professor Elias Manolakos, in the academic year 2023-2024.

## Cloning the Repository

```sh
git clone https://github.com/GiatrasKon/Gastric-Cancer-scRNAseq-Analysis.git
```

## Contributors
- [Konstantinos Giatras](https://github.com/GiatrasKon)
- [Olympia Tsiomou](https://github.com/otsiomou)

## Package Dependencies

Ensure you have the following packages installed:

- pandas
- IPython
- matplotlib
- seaborn
- numpy
- anndata
- scanpy
- scipy
- sklearn
- umap-learn
- gprofiler-official

Install dependencies using:

```sh
pip install pandas ipython matplotlib seaborn numpy anndata scanpy scipy scikit-learn umap-learn gprofiler-official
```

## Class Description

The `GCSingleCellAnalysis` class (located in `src/codebase.py`) provides a comprehensive workflow for preprocessing, dimensionality reduction, clustering, and functional annotation of single-cell RNA-seq data.

## Running the Analysis

1. **Preprocessing and Normalization:**
    ```python
    from src.codebase import GCSingleCellAnalysis
    analysis = GCSingleCellAnalysis('data/GSE158631_count.csv')
    analysis.preprocess_adata()
    analysis.normalize_adata()
    analysis.filter_adata()
    analysis.prepare_adata()
    ```

2. **Dimensionality Reduction:**
    ```python
    analysis.perform_pca()
    analysis.prepare_pca_reduced_adata()
    analysis.plot_pca()
    analysis.perform_tsne()
    analysis.plot_tsne()
    analysis.perform_umap()
    analysis.plot_umap()
    ```

3. **Clustering:**
    ```python
    methods = ['gmm', 'average_link', 'ward', 'spectral', 'louvain', 'leiden']
    results_pca = analysis.cluster_and_evaluate(methods, embeddings=['X_pca'])
    results_tsne = analysis.cluster_and_evaluate(methods, embeddings=['X_tsne'])
    results_umap = analysis.cluster_and_evaluate(methods, embeddings=['X_umap'])
    combined_results = {'PCA': results_pca, 't-SNE': results_tsne, 'UMAP': results_umap}
    results_df = analysis.create_results_dataframe(combined_results)
    analysis.plot_clustering_evaluation(results_df)
    ```

4. **Post-Clustering Analysis:**
    ```python
    analysis.analyze_and_plot_markers(group_key='average_link_X_umap', n_genes=10)
    go_annotations = analysis.fetch_go_annotations(group_key='average_link_X_umap', n_genes=10)
    analysis.print_go_annotations(go_annotations)
    ```

## Notebook

For a detailed step-by-step analysis, refer to the Jupyter notebook:
- `notebooks/GC_scRNAseq_data_analysis.ipynb`

Images of the results of the analysis can be found in the `images` directory.

## Documentation

Refer to the `documents` directory for the project proposal, presentation, report, and the original authors' paper.

---