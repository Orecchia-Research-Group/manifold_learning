import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances as euclid
import scanpy as sc

def get_sct_sparse():
	ILC = sc.read("data/sct.h5ad")
	return ILC.layers["norm_data"]

def get_sct_var_sparse():
	ILC = sc.read("data/sct_variable.h5ad")
	return ILC.layers["norm_data"]

def get_PCA_coord():
	ILC = sc.read("data/sct.h5ad")
	return ILC.obsm["X_pca"]

def get_PCA_coord_var():
	ILC_var = sc.read("data/sct_variable.h5ad")
	return ILC_var.obsm["X_pca"]

def get_Diff_coord():
	ILC = sc.read("data/sct.h5ad")
	return ILC.obsm["X_diffmap"]

def get_Diff_coord_var():
	ILC_var = sc.read("data/sct_variable.h5ad")
	return ILC_var.obsm["X_diffmap"]

def get_draw_graph_fa():
	ILC = sc.read("data/sct.h5ad")
	return ILC["X_draw_graph_fa"]

def get_draw_graph_fa_var():
	ILC_var = sc.read("data/sct_variable.h5ad")
	return ILC_var["X_draw_graph_fa"]

def get_UMAP():
	ILC = sc.read("data/sct.h5ad")
	return ILC["umap_cell_embeddings"]

def get_UMAP_var():
	ILC_var = sc.read("data/sct_variable.h5ad")
	return ILC_var["umap_cell_embeddings"]

def get_Gene_list():
	try:
		Gene_list = pd.read_csv("data/Gene_list_ILC.csv", sep=",", header = 0)
		Gene_list = np.array(Gene_list)
		Gene_list = Gene_list[:, 0]
	except FileNotFoundError:
		ILC_var = sc.read("data/sct_variable.h5ad")
		Variable = ILC_var.var
		Variable["Selected"].to_csv('data/Gene_list_ILC.csv', sep=',')

		Gene_list = pd.read_csv("data/Gene_list_ILC.csv", sep=",", header = 0)
		Gene_list = np.array(Gene_list)
		Gene_list = Gene_list[:, 0]
	return Gene_list

def get_Cell_list():
	try:
		Cell_list = pd.read_csv("data/Cell_list_ILC.csv", sep=",", header = 0)
		Cell_list = np.array(Cell_list)
		Cell_list = Cell_list[:,0]
	except FileNotFoundError:
		ILC_var = sc.read("data/sct_variable.h5ad")
		Obs = ILC_var.obs
		O = Obs["ilc2_ilc3"].to_frame()
		O.to_csv('data/Cell_list_ILC.csv', sep=',')

		Cell_list = pd.read_csv("data/Cell_list_ILC.csv", sep=",", header = 0)
		Cell_list = np.array(Cell_list)
		Cell_list = Cell_list[:,0]
	return Cell_list

def get_cells_of_interest():
	"""
	Get normalized gene expression of cells of interest
	"""
	try:
		ILCs_reads = pd.read_csv("data/ILCs_reads.csv", sep=",", header = None) #Normalized gene expression of cells of interest
		ILCs_index = pd.read_csv("data/ILCs_index.csv", sep=",", header = None)
		index_cells = pd.read_csv("data/index_cells.csv", sep=",", header=None)

		return ILCs_reads, ILCs_index, index_cells

	except FileNotFoundError:
		raise FileNotFoundError("To generate these CSVs, run ilc_data/cells_of_interest.py from the top directory.")

def get_dist_mat():
	try:
		return np.load("data/dist_mat.npy")

	except FileNotFoundError:
		print("Generating dist_mat...")
		points = get_sct_sparse().todense()
		dist_mat = euclid(points)
		np.save("data/dist_mat.npy", dist_mat)
		return dist_mat

def get_dist_mat_var():
	try:
		return np.load("data/dist_mat_var.npy")

	except FileNotFoundError:
		print("Generating dist_mat_var...")
		ILCs_reads, _, _ = get_cells_of_interest()
		ILCs = np.array(ILCs_reads)
		dist_mat = euclid(ILCs)
		np.save("data/dist_mat_var.npy", dist_mat)
		return dist_mat

def get_index_to_gene():
	ILC = sc.read("data/sct_variable.h5ad")
	ILC_df = ILC.to_df()
	return dict(enumerate(ILC_df.columns))
