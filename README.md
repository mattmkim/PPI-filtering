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
The function `cluster_atoms` takes in an integer representing the bin to cluster. Using Scipy, a `KDTree` was creating using the positions of the atoms in the specified bin, and the `query()` method was called to get the nearest neighbors of each atom. The nearest neighbors of each atom were traversed, and intersections between nearest neighbors of atoms were found to determine clusters. Result of clustering shown below (different colors represent different bins).

![vmdscene](https://user-images.githubusercontent.com/43687112/63386879-093a1400-c372-11e9-8e45-26810e6ea860.png)

Various adaptions of this algorithm were created in the _clustering_culm.py_ file. The `cluster_atoms_under()` and `cluster_atoms_over()` functions cluster atoms under and over a specified bin number respectively. The `clustering_info_under()` and `clustering_info_over()` functions provide information regarding the number of clusters, distribution of cluster sizes, as well as the indices of atoms that are a part of clusters of size 1, 2, or 3; this information would be used in the naive filtering approach. 

## Filtering 

## Naive Approach

The naive approach to filtering utilizes the aforementioned clustering algorithm. Visualizations of true and false positive atoms showed that false positives can be found in small clusters; thus, by determining the location of atoms in small clusters (sizes 1, 2 and 3), these atoms can be filtered out of the predictions of the PPI to improve the accuracy of PPI prediction. 

Data from simulations that indicated which atoms were predicted to be a part of the PPI when the protein was exposed to different values of a potential was used to determine the indices of true and false positives, using the `false_positives()` and `true_positives()` functions. For each value of potential, the method `clustering_info_pred_interface_atoms()` was called, which internally calls `cluster_pred_interface_atoms()`. This method clusters atoms that are predicted to be a part of the PPI at a specified potential value. By iterating through the clusters found, the atoms that are a part of clusters of sizes 1, 2 or 3 were found. 

Next, at each potential value where a prediction was made, intersections between the atoms considered as true and false positives and the atoms that were to be filtered out were removed, and the "new" true and false positives were stored in `FP_dict_filter` and `TP_dict_filter`. 

```
FP_dict_filter = {}
TP_dict_filter = {}

for i in range(0, 404, 4):
	print(i)
	num_clusters, cluster_dict, singleton_ind, doubles_ind, triples_ind = clustering_info_pred_interface_atoms(i)
	if len(singleton_ind) == 0:
		FP_dict_filter[i] = FP_dict[i]
		TP_dict_filter[i] = TP_dict[i]
	else:
		# for singleton clusters
		new_ind_FP = list(set(FP_dict[i]) - set(singleton_ind))
		new_ind_TP = list(set(TP_dict[i]) - set(singleton_ind))

		# for clusters with two atoms 
		new_ind_FP = list(set(new_ind_FP) - set(doubles_ind))
		new_ind_TP = list(set(new_ind_TP) - set(doubles_ind))

		# for clusters with three atoms
		new_ind_FP = list(set(new_ind_FP) - set(triples_ind))
		new_ind_TP = list(set(new_ind_TP) - set(triples_ind))

		FP_dict_filter[i] = new_ind_FP
		TP_dict_filter[i] = new_ind_TP
```
To measure the performance of this filter, a pre-filter and post-filter ROC curve was plotted, and the area under the curve was recorded.

![ROC](https://user-images.githubusercontent.com/43687112/63386764-d1cb6780-c371-11e9-97ea-150a6a39283b.png)

The filter was able to only slightly improve the accuracy of the prediction, which was expected. At low values of potential, the occurences of false positives were few and found in small clusters; however, as the potential grew, both the number and cluster size of false positives grew as well, and the filter was unable to account for large clusters of false positives. Further, true positives were also being filtered out, and due to the relatively low count of true positives, this had a significant effect on the area under the ROC curve.

The next approach taken was to determine if there was a set of atoms that could be filtered out at _every_ potential value. The rationale behind this was that this would limit the number of true positives being filtered out. In this approach, at each potential value, clusters of size 1, 2, and 3 among the atoms that were predicted to be a part of the PPI were found; next, those atoms were filtered out of the PPI prediction for every potential value, and an ROC curve was plotted. The "best" set of atoms yielded the largest area under the curve. A plot for the area under the curve versus potential value is shown below. 

![ROC_AREA](https://user-images.githubusercontent.com/43687112/63461565-e584d580-c426-11e9-8085-a96eca3717b4.png)

The optimal set of atoms was found from clusters of size 1, 2, or 3 at a potential value of 0.96. This set of atoms was filtered out from predictions at all other potential values, and a ROC curve was constructed.

![ROC_key](https://user-images.githubusercontent.com/43687112/63461813-7e1b5580-c427-11e9-8964-3fb54d870ddb.png)

Again, the performance of the filter was not very good. While this approach succeeded in avoiding filtering out true positives, it was only able to filter out 6 false positive atoms. Furthermore, this approach was not consistent; when applied to other proteins, the optimal set of atoms to filter out was no atoms - the filter was filtering out true positive atoms. 

## Machine Learning Approach

The naive approach to the filtering problem had some success, but ultimately could not serve as a comprehensive solution - there were false positive atoms that were in contact with true positive atoms, which the aforementioned filter could not filter out. Data for five proteins were processed into a Pandas dataframe and was used to train and test a neural network created using the Pytorch framework. Only atoms that were predicted to be a part of the PPI were used so that the model could classify atoms as either a true or false positive. 

## Features

The feature vector has 62 elements and contains information for each atom in a protein regarding its atom type, residue type, the potential value (affects the number of dewetted nearest neighbors), number of nearest neighbors, number of dewetted nearest neighbors, and if it is a true positive or false positive atom.

### Atom and Residue Type

Each atom in a protein was labeled with an atom and residue type. Since these are nominal variables, these values were encoded using the Pandas `get_dummies` method. 

```
# encode nominal variables
df = pd.get_dummies(df, columns=['atom_name'], prefix=['atom_name'])
df = pd.get_dummies(df, columns=['residue_name'], prefix=['residue_name'])
```

### Number of Nearest Neighbors and Dewetted Nearest Neighbors

Since data for five different proteins was being used, an atom's x, y, and z coordinate can not be used. The `get_num_nn()` and `get_num_dewet_nn()` functions return a list that contains the number of nearest neighbors and dewetted nearest neighbors for every atom; these two features would give information regarding an atoms relative position. 

Finding the number of nearest neighbors for each atom was straightforward. All atoms in each protein were clustered using the same aforementioned method. The clusters were then iterated through to determine the number of nearest neighbors of each atom (the size of the cluster an atom is in).

```
# list to contain the number of nearest neighbors for each atom 
nn_list = []

# positions is a list of 3-tuples representing each atoms coordinates
for i in range(0, len(positions)):
	for j in range(0, len(final_clusters)):
	
		# atoms in final_clusters are represented as integers, in the same order as they appear in positions
		if i in final_clusters[j]:
			nn_list.append(len(final_clusters[j]))
			break
			
# dataframe contains data for each atom at each potential value; since number of nearest neighbors doesn't depend on potential value, simply can extend the list by 51 times. 	
nn_list = nn_list * 51

return nn_list
```
Determining 


