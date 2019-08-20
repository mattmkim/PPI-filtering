# PPI-filtering 
## Overview

Goals of research were to develop a new method for clustering atoms on proteins and to improve the accuracy of protein-protein interface (PPI) predictions. Clustering was dependent on both the positions of atoms and their susceptibility to dewetting in the presence of an unfavorable potential. Two approaches were taken for the filtering problem - a naive approach that utilized clustering, as well as a more comprehensive approach that utilized neural networks.

### Clustering 

Clustering algorithm was created in _clustering.py_ file. The algorithm first bins all the atoms in a given protein based on a value that indicates their susceptiblity to dewetting. 
```
# num indicates number of bins to create
phi_star_bins = np.linspace(0, max(phi_star), num=101)
inds = np.digitize(phi_star, phi_star_bins)
```

