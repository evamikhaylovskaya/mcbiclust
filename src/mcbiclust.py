# mcbiclust.py
import numpy as np
from gene_seed import select_initial_seed_genes
from find_seed_sample import find_seed_bicluster
from pruning import prune_bicluster_genes
from sample_sort import extend_bicluster_samples_fast
from correlation_vector import cv_eval
from pca import pc1_vec_fun
from thresholding import threshold_bic, pc1_align
from fork import  fork_classifier
from multi_run import run_multiple, multi_sample_sort_prep, average_corvec_per_cluster
from silhouette import silhouette_clust_groups, plot_silhouette


class MCbiclust:
    """
    Python reimplementation of the MCbiclust biclustering algorithm.

    Runs the full pipeline:
      1. FindSeed       — greedy search for seed samples
      2. Pruning        — HclustGenesHiCor gene pruning
      3. SampleSort     — rank all samples by alpha
      4. CVEval         — compute correlation vector
      5. PC1VecFun      — compute PC1 values
      6. ThresholdBic   — define bicluster genes and samples
      7. PC1Align       — align PC1 sign
      8. ForkClassifier — classify upper/lower fork

    Parameters
    ----------
    n_samples   : int   seed size (default 10)
    iterations  : int   FindSeed iterations (default 1000)
    n_genes     : int   initial gene set size (default 1000)
    splits      : int   hierarchical groups for pruning (default 8)
    samp_sig    : float ThresholdBic strictness 0-1 (default 0.05)
    random_state: int   random seed (default 42)
    """

    def __init__(self, n_samples=10, iterations=1000, n_genes=1000,
                 splits=8, samp_sig=0.05, random_state=42):
        self.n_samples    = n_samples
        self.iterations   = iterations
        self.n_genes      = n_genes
        self.splits       = splits
        self.samp_sig     = samp_sig
        self.random_state = random_state

        # results — set after fit()
        self.seed_          = None
        self.seed_alpha_    = None
        self.pruned_genes_  = None
        self.ranked_        = None
        self.sample_alphas_ = None
        self.cor_vec_       = None
        self.pc1_           = None
        self.bic_genes_     = None
        self.bic_samps_     = None
        self.fork_          = None
        self.av_corvec_signed_ = None
        self.improvements_ = None
        self.history_ = None
        self.history_iterations_ = None
        self.kept_local_idx_ = None
        self.cluster_labels_ = None
        self.hi_cor_values_ = None
        self.group_alphas_ = None
        self.best_idx_ = None
        self.picked_ = None

    def fit(self, dataset, gene_set=None):
        """
        Run the full MCbiclust pipeline on a single run.

        Parameters
        ----------
        dataset  : ExpressionDataset
        gene_set : np.ndarray  optional gene indices to use
                               if None, selects random n_genes genes

        Returns
        -------
        self
        """
        X = dataset.X

        # initial gene set
        if gene_set is None:
            gene_set = select_initial_seed_genes(
                X=X,
                n_genes=self.n_genes,
                biological_gene_set=np.arange(X.shape[0]),
                random_state=self.random_state
            )
        self.gene_set_ = gene_set
        
        # FindSeed
        (
            self.seed_,
            self.seed_alpha_,
            self.improvements_,
            self.history_,
            self.history_iterations_,
        ) = find_seed_bicluster(
            X=X,
            gene_set=gene_set,
            n_samples=self.n_samples,
            iterations=self.iterations,
            random_state=self.random_state,
            log_improvements=True,
            print_every_improvement=False,
            print_every_iter=False,
            verbose=False,
        )


        # HclustGenesHiCor
        (
            self.kept_local_idx_,
            self.pruned_genes_,
            self.cluster_labels_,
            self.hi_cor_values_,
            self.group_alphas_,
        ) = prune_bicluster_genes(
            X=X,
            gene_set=gene_set,
            sample_set=self.seed_,
            n_groups=self.splits,
            plot_dendrogram=False,
        )

        # SampleSort
        self.ranked_, self.sample_alphas_, self.picked_ = extend_bicluster_samples_fast(
            X=X,
            gene_set=self.pruned_genes_,
            initial_sample_set=self.seed_,
            progress_every=50,
            print_only_improvements=False,
            max_rank=None,
        )

        # CVEval
        self.cor_vec_, self.best_idx_ = cv_eval(
            X_part=X[self.pruned_genes_],
            X_all=X,
            seed=self.seed_,
            splits=10,
            gene_names=getattr(dataset, "genes", None),
        )

        # PC1VecFun
        self.pc1_ = pc1_vec_fun(
            X=X,
            gene_set=self.pruned_genes_,
            seed_sort=self.ranked_,
            n=self.n_samples,
            align_sign=False
        )

        # ThresholdBic
        self.bic_genes_, self.bic_samps_ = threshold_bic(
            cor_vec=self.cor_vec_,
            sort_order=self.ranked_,
            pc1=self.pc1_,
            samp_sig=self.samp_sig,
            sample_names=getattr(dataset, "samples", None),
        )

        # PC1Align
        self.pc1_ = pc1_align(
            gem=X,
            pc1=self.pc1_,
            sort_order=self.ranked_,
            cor_vec=self.cor_vec_,
            bic=(self.bic_genes_, self.bic_samps_),
        )

        # ForkClassifier
        self.fork_ = fork_classifier(
            pc1=self.pc1_,
            samp_num=len(self.bic_samps_),
        )

        return self

    def summary(self):
        """Print bicluster summary."""
        n_bic   = len(self.bic_samps_) if self.bic_samps_ is not None else 0
        n_upper = int(np.sum(self.fork_ == "Upper")) if self.fork_ is not None else 0
        n_lower = int(np.sum(self.fork_ == "Lower")) if self.fork_ is not None else 0

        print(f"MCbiclust Results")
        print(f"  Seed alpha:        {self.seed_alpha_:.4f}")
        print(f"  Pruned genes:      {len(self.pruned_genes_)}")
        print(f"  Bicluster genes:   {self.bic_genes_.sum()}")
        print(f"  Bicluster samples: {n_bic}")
        print(f"  Upper fork:        {n_upper}")
        print(f"  Lower fork:        {n_lower}")
        
    def to_sample_dataframe(self, dataset):
        from metadata_analysis import build_sample_results_dataframe

        if self.ranked_ is None or self.pc1_ is None or self.fork_ is None:
            raise ValueError("Run fit() first.")

        return build_sample_results_dataframe(
            ranked_samples=self.ranked_,
            sample_names=dataset.samples,
            pc1_vec=self.pc1_,
            fork_status=self.fork_,
        )
        
    def merge_sample_metadata(self, dataset, metadata_df):
        from metadata_analysis import (
            build_sample_results_dataframe,
            merge_sample_metadata,
        )

        if self.ranked_ is None or self.pc1_ is None or self.fork_ is None:
            raise ValueError("Run fit() first.")

        results_df = build_sample_results_dataframe(
            ranked_samples=self.ranked_,
            sample_names=dataset.samples,
            pc1_vec=self.pc1_,
            fork_status=self.fork_,
        )

        return merge_sample_metadata(results_df, metadata_df)
    
    
    def run_go_enrichment(self, gene_names, gaf_path,
                      top_n=20, min_genes=5,
                      alternative="greater", save_path=None):
        from enrichment import go_enrichment_pipeline

        if self.cor_vec_ is None:
            raise ValueError("Run fit() first.")

        return go_enrichment_pipeline(
            cor_vec=self.cor_vec_,
            gene_names=gene_names,
            gaf_path=gaf_path,
            top_n=top_n,
            min_genes=min_genes,
            alternative=alternative,
            save_path=save_path,
        )
        
    def plot_pc1_by_category(self, dataset, metadata_df, category_col, title=None):
        from metadata_analysis import (
            build_sample_results_dataframe,
            merge_sample_metadata,
            plot_pc1_by_category,
        )

        results_df = build_sample_results_dataframe(
            ranked_samples=self.ranked_,
            sample_names=dataset.samples,
            pc1_vec=self.pc1_,
            fork_status=self.fork_,
        )

        merged_df = merge_sample_metadata(results_df, metadata_df)

        plot_pc1_by_category(
            merged_df,
            category_col=category_col,
            title=title,
        )


class MCbiclustMulti:
    """
    Run MCbiclust multiple times to find all distinct biclusters.

    Parameters
    ----------
    n_runs        : int   number of independent runs (default 20)
    n_genes       : int   initial gene set size (default 1000)
    n_samples     : int   seed size (default 10)
    iterations    : int   FindSeed iterations per run (default 500)
    splits        : int   hierarchical groups (default 8)
    samp_sig      : float ThresholdBic strictness (default 0.8)
    max_clusters  : int   max clusters for silhouette (default 10)
    top_genes_num : int   genes used in MultiSampleSortPrep (default 1000)
    """

    def __init__(self, n_runs=20, n_genes=1000, n_samples=10,
                 iterations=500, splits=8, samp_sig=0.05,
                 max_clusters=10, top_genes_num=1000, cv_splits = 10):
        self.n_runs        = n_runs
        self.n_genes       = n_genes
        self.n_samples     = n_samples
        self.iterations    = iterations
        self.splits        = splits
        self.samp_sig      = samp_sig
        self.max_clusters  = max_clusters
        self.top_genes_num = top_genes_num
        self.cv_splits = cv_splits
        
        # results
        self.cor_vec_mat_    = None
        self.seeds_list_     = None
        self.cluster_groups_ = None
        self.av_corvec_      = None
        self.biclusters_     = []

    def fit(self, dataset):
        """
        Run full multi-run MCbiclust pipeline.

        Parameters
        ----------
        dataset : ExpressionDataset

        Returns
        -------
        self
        """
        X = dataset.X

        # multiple runs
        self.cor_vec_mat_, self.seeds_list_ = run_multiple(
            X          = X,
            n_runs     = self.n_runs,
            n_genes    = self.n_genes,
            n_samples  = self.n_samples,
            iterations = self.iterations,
            splits     = self.splits,
            cv_splits= self.cv_splits
        )
        

        # silhouette clustering
        self.cluster_groups_ = silhouette_clust_groups(
            cor_vec_mat  = self.cor_vec_mat_,
            max_clusters = self.max_clusters,
            plots        = True
        )

        plot_silhouette(
            cor_vec_mat=self.cor_vec_mat_,
            cluster_groups=self.cluster_groups_,
            title="Silhouette plot of biclusters"
        )
        
        self.av_corvec_, self.av_corvec_signed_ = average_corvec_per_cluster(
            self.cor_vec_mat_,
            self.cluster_groups_,
        )

        # MultiSampleSortPrep
        top_mats, top_seeds = multi_sample_sort_prep(
            X             = X,
            av_corvec     = self.av_corvec_,
            top_genes_num = self.top_genes_num,
            groups        = self.cluster_groups_,
            initial_seeds = self.seeds_list_
        )

        # full pipeline per cluster
        self.biclusters_ = []
        for k in range(len(self.cluster_groups_)):
            print(f"\nFitting cluster {k+1}...")
            mc = MCbiclust(
                n_samples = self.n_samples,
                splits    = self.splits,
                samp_sig  = self.samp_sig
            )

            # inject pre-computed results
            mc.seed_    = top_seeds[k]
            gene_set_k  = np.argsort(np.abs(self.av_corvec_[k]))[::-1][:self.top_genes_num]
            
            (
                mc.kept_local_idx_,
                mc.pruned_genes_,
                mc.cluster_labels_,
                mc.hi_cor_values_,
                mc.group_alphas_,
            ) = prune_bicluster_genes(
                X=X,
                gene_set=gene_set_k,
                sample_set=top_seeds[k],
                n_groups=self.splits,
                plot_dendrogram=False,
            )

            mc.ranked_, mc.sample_alphas_, _ = extend_bicluster_samples_fast(
                X=top_mats[k],
                gene_set=np.arange(top_mats[k].shape[0]),
                initial_sample_set=top_seeds[k],
                progress_every=50,
                print_only_improvements=False,
                max_rank=None,
            )
            
            mc.cor_vec_ = self.av_corvec_[k].copy()
            mc.best_idx_ = None

           
            
            mc.pc1_ = pc1_vec_fun(
                X=top_mats[k],
                gene_set=np.arange(top_mats[k].shape[0]),
                seed_sort=mc.ranked_,
                n=self.n_samples,
                align_sign=False,
            )

           

            mc.bic_genes_, mc.bic_samps_ = threshold_bic(
                cor_vec    = mc.cor_vec_,
                sort_order = mc.ranked_,
                pc1        = mc.pc1_,
                samp_sig   = self.samp_sig,
                sample_names=getattr(dataset, "samples", None),
            )

            mc.pc1_ = pc1_align(
                gem=X,
                pc1=mc.pc1_.copy(),
                sort_order=mc.ranked_,
                cor_vec=mc.cor_vec_,
                bic=(mc.bic_genes_, mc.bic_samps_),
            )

            n_bic = len(mc.bic_samps_) if mc.bic_samps_ is not None else 0
            if n_bic >= 2:
                mc.fork_ = fork_classifier(mc.pc1_, samp_num=n_bic).copy()
            else:
                mc.fork_ = np.full(len(mc.ranked_), "None", dtype=object)

            self.biclusters_.append(mc)

        return self

    def summary(self):
        """Print summary of all biclusters."""
        print(f"\nMCbiclustMulti Results")
        print(f"  Runs:              {self.n_runs}")
        print(f"  Distinct clusters: {len(self.cluster_groups_) if self.cluster_groups_ is not None else 0}")
        print()
        print(f"{'Cluster':<10}{'Genes':<10}{'Samples':<10}"
              f"{'Upper':<10}{'Lower':<10}")
        print(f"{'='*50}")
        for k, mc in enumerate(self.biclusters_):
            n_genes = int(mc.bic_genes_.sum()) if mc.bic_genes_ is not None else 0
            n_samps = len(mc.bic_samps_) if mc.bic_samps_ is not None else 0
            n_upper = int(np.sum(mc.fork_ == "Upper")) if mc.fork_ is not None else 0
            n_lower = int(np.sum(mc.fork_ == "Lower")) if mc.fork_ is not None else 0
            print(f"{k+1:<10}{n_genes:<10}"
                  f"{n_samps:<10}{n_upper:<10}{n_lower:<10}")
            
            
    
    