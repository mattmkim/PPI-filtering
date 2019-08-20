# PPI-filtering 
## Overview

Goals of research were to develop a new method for clustering atoms on proteins and to improve the accuracy of protein-protein interface (PPI) predictions. Clustering was dependent on both the positions of atoms and their susceptibility to dewetting in the presence of an unfavorable potential. Two approaches were taken for the filtering problem - a naive approach that utilized clustering, as well as a machine learning approach that utilized neural networks.

## Clustering 

Clustering algorithm was created in _clustering.py_ file. The algorithm first bins all the atoms in a given protein based on a value that indicates their susceptiblity to dewetting. 
```
# num indicates number of bins to create

phi_star_bins = np.linspace(0, max(phi_star), num=50)
inds = np.digitize(phi_star, phi_star_bins)
```
The function _cluster_atoms_ takes in an _int_ representing the bin to cluster. Using Scipy, a _KDTree_ was creating using the positions of the atoms in the specified bin, and the _query()_ method was called to get the nearest neighbors of each atom. The nearest neighbors of each atom were traversed, and intersections between nearest neighbors of atoms were found to determine clusters. Result of clustering shown below (different colors represent different bins).

![Alt text](/Users/matthew/Desktop/Summer Research 2019/vmdscene.tga?raw=true "Title")

Various adaptions of this algorithm were created in the _clustering_culm.py_ file. The _cluster_atoms_under_ and _cluster_atoms_over_ methods cluster atoms under and over a specified bin number respectively. The _clustering_info_under_ and _clustering_info_over_ methods provide information regarding the number of clusters, distribution of cluster sizes, as well as the indices of atoms that are a part of clusters of size 1, 2, or 3; this information would be used in the naive filtering approach. 

## Filtering 

### Naive Approach

The naive approach to filtering utilizes the aforementioned clustering algorithm. Visualizations of true and false positive atoms showed that false positives can be found in small clusters. Data from simulations that indicated which atoms were predicted to be a part of the PPI when the protein was exposed to different values of a potential was used to determine the indices of true and false positives, using the _false_positives_ and _true_positives_ methods. For each value of potential, the method _clustering_info_pred_interface_atoms_ was called, which internally calls _cluster_pred_interface_atoms_. This method clusters atoms that are predicted to be a part of the PPI at a specified potential value. 




