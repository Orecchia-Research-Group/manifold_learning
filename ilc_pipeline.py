import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from numpy import asarray
from numpy import savetxt
import matplotlib.colors as mcolors
import matplotlib.cm
import matplotlib.patches as mpatches
import umap
import matplotlib as mpl
import scanpy as sc

from ilc_data.ilc_loader import *

# Read in data
# Select either the whole sc transformed dataset or just the variable genes

ILC_var = sc.read("sct_variable.h5ad")
ILC = sc.read("sct.h5ad")

PCA_coord = get_PCA_coord_var()
Diff_coord = get_Diff_coord_var()

#norm_reads_sparse = ILC_var.layers["norm_data"]
norm_reads_sparse = ILC.layers["norm_data"]
norm_reads = pd.DataFrame(norm_reads_sparse.toarray())
print("Norm reads: "+repr(norm_reads.shape))

UMAP = get_UMAP_var()
Diff = get_draw_graph_fa_var()

# Selecting all ILC2 and ILC3 cells
Gene_list = get_Gene_list()
Cell_list = get_Cell_list()

# Collecting cells of interest

# In this model of psoriasis, tissue resident ILC2s are reprogramed to ILC3s.
# For analysis we are selecting for cells likely to be ILC2s or ILC3s on the basis of
# previously characterized populations.

#We have four labeled groups of cells that we want to collect together
ILC2_ILC3 = ILC_var.obs["ilc2_ilc3"]
ILC3_Q = ILC_var.obs["quiescent_ilc3"]
ILC2_Q = ILC_var.obs["ilc2_quiescent"]
cloud_ILC3 = ILC_var.obs["cloud_ilc3"]

df = pd.DataFrame(dict(ILC2_ILC3 = ILC2_ILC3, ILC3_Q = ILC3_Q, ILC2_Q = ILC2_Q, cloud_ILC3 = cloud_ILC3))
cells = df.shape[0]

#binary yes ir no for if the cell is in our group
in_transition = np.zeros((cells, 1))

#Value of group - 0 if NA - over all cells
allcell_ILC2_ILC3 = np.zeros((cells, 1))
allcell_ILC3_Q = np.zeros((cells, 1))
allcell_ILC2_Q = np.zeros((cells, 1))
allcell_cloud_ILC3 = np.zeros((cells, 1))

#Cell labels - 0 if NA in all groups - over all cells
index_cells = np.zeros((cells, 1))
index_cells = index_cells.astype(str)

#Value of group - only cells in at least one group
val_ILC2_ILC3 = np.zeros((3807, 1))
val_ILC3_Q = np.zeros((3807, 1))
val_ILC2_Q = np.zeros((3807, 1))
val_cloud_ILC3 = np.zeros((3807, 1))

#Names of cells included in the value variables
index_val = np.array([])

#Variable for initially counting the number of cells in the groups of interest
number = 0


#Loops through the different groups so that we collect all cells with a value in at least one of the groups
for i in range(0,cells):
    if np.isnan(df["ILC2_ILC3"][i]):

        if np.isnan(df["ILC3_Q"][i]):

            if np.isnan(df["ILC2_Q"][i]):

                if np.isnan(df["cloud_ILC3"][i]):

                    in_transition[i,0] = 0

                    allcell_ILC2_ILC3[i,0] = 0
                    allcell_ILC3_Q[i,0] = 0
                    allcell_ILC2_Q[i,0] = 0
                    allcell_cloud_ILC3[i,0] = 0

                    index_cells[i,0] = "0"

                else:
                    in_transition[i,0] = 1

                    allcell_ILC2_ILC3[i,0] = df["ILC2_ILC3"][i]
                    allcell_ILC3_Q[i,0] = df["ILC3_Q"][i]
                    allcell_ILC2_Q[i,0] = df["ILC2_Q"][i]
                    allcell_cloud_ILC3[i,0] = df["cloud_ILC3"][i]

                    index_cells[i,0] = Cell_list[i]

                    val_ILC2_ILC3[number] = df["ILC2_ILC3"][i]
                    val_ILC3_Q[number] = df["ILC3_Q"][i]
                    val_ILC2_Q[number] = df["ILC2_Q"][i]
                    val_cloud_ILC3[number] = df["cloud_ILC3"][i]

                    index_val = np.append(index_val , df["ILC2_ILC3"].index[i])

                    number += 1
            else:
                in_transition[i,0] = 1

                allcell_ILC2_ILC3[i,0] = df["ILC2_ILC3"][i]
                allcell_ILC3_Q[i,0] = df["ILC3_Q"][i]
                allcell_ILC2_Q[i,0] = df["ILC2_Q"][i]
                allcell_cloud_ILC3[i,0] = df["cloud_ILC3"][i]

                index_cells[i,0] = Cell_list[i]

                val_ILC2_ILC3[number] = df["ILC2_ILC3"][i]
                val_ILC3_Q[number] = df["ILC3_Q"][i]
                val_ILC2_Q[number] = df["ILC2_Q"][i]
                val_cloud_ILC3[number] = df["cloud_ILC3"][i]

                index_val = np.append(index_val , df["ILC2_ILC3"].index[i])

                number += 1

        else:
            in_transition[i,0] = 1

            allcell_ILC2_ILC3[i,0] = df["ILC2_ILC3"][i]
            allcell_ILC3_Q[i,0] = df["ILC3_Q"][i]
            allcell_ILC2_Q[i,0] = df["ILC2_Q"][i]
            allcell_cloud_ILC3[i,0] = df["cloud_ILC3"][i]

            index_cells[i,0] = Cell_list[i]

            val_ILC2_ILC3[number] = df["ILC2_ILC3"][i]
            val_ILC3_Q[number] = df["ILC3_Q"][i]
            val_ILC2_Q[number] = df["ILC2_Q"][i]
            val_cloud_ILC3[number] = df["cloud_ILC3"][i]

            index_val = np.append(index_val , df["ILC2_ILC3"].index[i])

            number += 1

    else:
        in_transition[i,0] = 1

        allcell_ILC2_ILC3[i,0] = df["ILC2_ILC3"][i]
        allcell_ILC3_Q[i,0] = df["ILC3_Q"][i]
        allcell_ILC2_Q[i,0] = df["ILC2_Q"][i]
        allcell_cloud_ILC3[i,0] = df["cloud_ILC3"][i]

        index_cells[i,0] = Cell_list[i]

        val_ILC2_ILC3[number] = df["ILC2_ILC3"][i]
        val_ILC3_Q[number] = df["ILC3_Q"][i]
        val_ILC2_Q[number] = df["ILC2_Q"][i]
        val_cloud_ILC3[number] = df["cloud_ILC3"][i]

        index_val = np.append(index_val , df["ILC2_ILC3"].index[i])

        number += 1

print(number)

# Save or read in matrix with cells of interest

np.savetxt("ILCs_reads.csv", reads, delimiter=',')
np.savetxt("ILCs_index.csv", index_val, delimiter=',', fmt="%s")
np.savetxt("index_cells.csv", index_cells, delimiter=',', fmt="%s")

ILCs_reads = pd.read_csv("ILCs_reads.csv", sep=",", header = None) #Normalized gene expression of cells of interest
ILCs_reads_values = np.array(ILCs_reads)

ILCs_index = pd.read_csv("ILCs_index.csv", sep=",", header = None)

# UMAP of selected cells

reducer = umap.UMAP()
embedding = reducer.fit_transform(ILCs_reads_values)
embedding.shape

#np.savetxt("UMAP_all_ILC_norm.csv", embedding, delimiter=',')
#embedding = pd.read_csv("UMAP_all_ILC_norm.csv", sep=",", header = None)
#embedding = np.array(embedding)

frame1 = plt.scatter(embedding[:,0], embedding[:,1], s= 5, c = val_ILC2_ILC3)
plt.title('All ILC UMAP - ILC2_ILC3 value')
#plt.xlim(-30000,25000)
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.ylabel('UMAP2')
plt.xlabel('UMAP1')
plt.show()

frame2 = plt.scatter(embedding[:,0], embedding[:,1], s= 5, c = val_ILC3_Q)
plt.title('All ILC UMAP - Quiescent ILC3 value')
#plt.xlim(-30000,25000)
frame2.axes.get_xaxis().set_ticks([])
frame2.axes.get_yaxis().set_ticks([])
plt.ylabel('UMAP2')
plt.xlabel('UMAP1')
plt.show()

frame3 = plt.scatter(embedding[:,0], embedding[:,1], s= 5, c = val_ILC2_Q)
plt.title('All ILC UMAP - Quiescent ILC2 value')
#plt.xlim(-30000,25000)
frame3.axes.get_xaxis().set_ticks([])
frame3.axes.get_yaxis().set_ticks([])
plt.ylabel('UMAP2')
plt.xlabel('UMAP1')
plt.show()

frame4 = plt.scatter(embedding[:,0], embedding[:,1], s= 5, c = val_cloud_ILC3)
plt.title('All ILC UMAP - Cloud ILC3 value')
#plt.xlim(-30000,25000)
frame4.axes.get_xaxis().set_ticks([])
frame4.axes.get_yaxis().set_ticks([])
plt.ylabel('UMAP2')
plt.xlabel('UMAP1')
plt.show()

# To look at individual genes on this UMAP
#To look at expression of individual genes on the UMAP we pull the index from our Gene_list

G = "Il4"
Gene_list = np.array(Gene_list)

for i in range(0,genes):
    if Gene_list[i] == G:
        gene_index = i

print(str(G)+' index: '+str(gene_index))

frame1 = plt.scatter(embedding[:,0], embedding[:,1], s= 5, c = ILCs_reads_values[:,1722])
plt.title('ILC2_ILC3 UMAP - Gzmb expression')
#plt.xlim(-30000,25000)
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.ylabel('UMAP2')
plt.xlabel('UMAP1')
plt.show()

frame1 = plt.scatter(embedding[:,0], embedding[:,1], s= 5, c = ILCs_reads_values[:,8])
plt.title('ILC2_ILC3 UMAP - Il17f expression')
#plt.xlim(-30000,25000)
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.ylabel('UMAP2')
plt.xlabel('UMAP1')
plt.show()

frame1 = plt.scatter(embedding[:,0], embedding[:,1], s= 5, c = ILCs_reads_values[:,2427])
plt.title('ILC2_ILC3 UMAP - Fos expression')
#plt.xlim(-30000,25000)
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.ylabel('UMAP2')
plt.xlabel('UMAP1')
plt.show()

frame1 = plt.scatter(embedding[:,0], embedding[:,1], s= 5, c = ILCs_reads_values[:,881])
plt.title('ILC2_ILC3 UMAP - Cxcl2 expression')
#plt.xlim(-30000,25000)
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.ylabel('UMAP2')
plt.xlabel('UMAP1')
plt.show()

frame1 = plt.scatter(embedding[:,0], embedding[:,1], s= 5, c = ILCs_reads_values[:,2019])
plt.title('ILC2_ILC3 UMAP - Il13 expression')
#plt.xlim(-30000,25000)
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.ylabel('UMAP2')
plt.xlabel('UMAP1')
plt.show()

frame1 = plt.scatter(embedding[:,0], embedding[:,1], s= 5, c = ILCs_reads_values[:,2018])
plt.title('ILC2_ILC3 UMAP - Il4 expression')
#plt.xlim(-30000,25000)
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.ylabel('UMAP2')
plt.xlabel('UMAP1')
plt.show()

### Preprocessing

center = np.argsort(val_ILC2_ILC3)[val_ILC2_ILC3.shape[0]//2]
print(center)

from manifold_utils.mSVD import eigen_plot, rapid_eigen_calc_from_dist_mat, eps_projection
from manifold_utils.iga import chakraborty_express, iga

ILCs_index = np.array(ILCs_index)

ILCs = ILCs_reads_values
N, d = ILCs.shape

dist_mat = np.zeros((N, N))
for j in range(0,N):
    for k in range(0,N):
        if j != k:
            dist_mat[j, k] = np.linalg.norm(ILCs[j, :] - ILCs[k, :])

#print(dist_mat.shape)

#savetxt('dist_mat_allILC.csv', dist_mat, delimiter=',')

dist_mat = pd.read_csv('dist_mat_allILC.csv', sep=",", header = None)
dist_mat = np.array(dist_mat )

#PCA distance
N, d = PCA_reads.shape

dist_mat_PCA = np.zeros((N, N))
for j in range(0,N):
    for k in range(0,N):
        if j != k:
            dist_mat_PCA[j, k] = np.linalg.norm(PCA_reads[j, :] - PCA_reads[k, :])

#Diffusion distance
N, d = Diff_reads.shape

dist_mat_Diff = np.zeros((N, N))
for j in range(0,N):
    for k in range(0,N):
        if j != k:
            dist_mat_Diff[j, k] = np.linalg.norm(Diff_reads[j, :] - Diff_reads[k, :])

### Applying preproccessing to matrix - ILC_gene_expression_values
center = 2500
radii, numPoints_list, eigval_list, eigvec_list = rapid_eigen_calc_from_dist_mat(ILCs, dist_mat, center)

rmin = radii[0]
rmax = radii[-1]

eigen_plot(eigval_list, numPoints_list)

print('Min: '+repr((21 - rmin)/0.01))
print('Max: '+repr((22 - rmin)/0.01))

evec = eigvec_list[269][:,0:2]
print(evec.shape)

evec_list = []

for i in range(0,100):
    evec_list.append( eigvec_list[218+i][:,0:2])
    
print(type(evec_list))

IGA = iga(evec_list)

IGA.shape

center = 2500
center_val = 2500
proj = eps_projection(ILCs,IGA,ILCs[center, :])

p = np.array(proj)

#savetxt('eps_projection_allILC.csv', p, delimiter=',')

in_ball = np.zeros(val_ILC2_ILC3.shape)
p_in_ball = np.zeros((1,p.shape[1]))
val_ILC2_ILC3_in_ball = np.array([])
val_ILC3_Q_in_ball = np.array([])
val_ILC2_Q_in_ball = np.array([])
val_cloud_ILC3_in_ball = np.array([])

for i in range(0,3807):
    if dist_mat[i, center_val] < 23.5:
        in_ball[i] = 1
        p_in_ball = np.append(p_in_ball, np.reshape(p[i,:], (1,2)), axis = 0)
        
        val_ILC2_ILC3_in_ball = np.append(val_ILC2_ILC3_in_ball, val_ILC2_ILC3[i])
        val_ILC3_Q_in_ball = np.append(val_ILC3_Q_in_ball, val_ILC3_Q[i])
        val_ILC2_Q_in_ball = np.append(val_ILC2_Q_in_ball, val_ILC2_Q[i])
        val_cloud_ILC3_in_ball = np.append(val_cloud_ILC3_in_ball, val_cloud_ILC3[i])
    else:
        in_ball[i] = 0
        
p_in_ball = p_in_ball[1:, :]
print(p_in_ball.shape)

frame1 = plt.scatter(p_in_ball[:,0], p_in_ball[:,1], s = 10, c = val_ILC2_ILC3_in_ball)
plt.title('Eps projection - ILC2_ILC3')
#plt.xlim(-0.1,0.1)
#plt.ylim(-0.1,0.1)
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.ylabel('Projection 2')
plt.xlabel('Projection 1')
plt.show()

frame2 = plt.scatter(p_in_ball[:,0], p_in_ball[:,1], s = 10, c = val_ILC3_Q_in_ball)
plt.title('Eps projection - Quiescent ILC3')
#plt.xlim(-0.1,0.1)
#plt.ylim(-0.1,0.1)
frame2.axes.get_xaxis().set_ticks([])
frame2.axes.get_yaxis().set_ticks([])
plt.ylabel('Projection 2')
plt.xlabel('Projection 1')
plt.show()

frame3 = plt.scatter(p_in_ball[:,0], p_in_ball[:,1], s = 10, c = val_ILC2_Q_in_ball)
plt.title('Eps projection - Quiescent ILC2')
#plt.xlim(-0.1,0.1)
#plt.ylim(-0.1,0.1)
frame3.axes.get_xaxis().set_ticks([])
frame3.axes.get_yaxis().set_ticks([])
plt.ylabel('Projection 2')
plt.xlabel('Projection 1')
plt.show()

frame4 = plt.scatter(p_in_ball[:,0], p_in_ball[:,1], s = 10, c = val_cloud_ILC3_in_ball)
plt.title('Eps projection - Cloud ILC3')
#plt.xlim(-0.1,0.1)
#plt.ylim(-0.1,0.1)
frame4.axes.get_xaxis().set_ticks([])
frame4.axes.get_yaxis().set_ticks([])
plt.ylabel('Projection 2')
plt.xlabel('Projection 1')
plt.show()
