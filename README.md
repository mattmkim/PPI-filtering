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
Determining the number of dewetted nearest neighbors was slighlty more complicated. For each potential value, the positions of atoms that were dewetted were put into a `KDTree`. If no atoms were dewetted at a potential value, all atoms at that potential value had 0 dewetted nearest neighbors. Thus, a list of zeros was concatenated to a runnning list containing the number of dewetted nearest neighbors, `dewet_nn_list`.

```
# only concerned about potential values between 0 and 2.00, at 0.04 increments
for i in range(0, 204, 4):
	filename = protein_name + ('/beta_phi_%03d/pred_contact_mask.dat' % (i))
	
	# created numpy array containing 1, representing that the atom was dewetted, or 0, representing that the atom was
	# not dewetted 
	contact_pred = np.loadtxt(filename)
	contact_pred_positions = positions[np.where(contact_pred == 1)[0].tolist()] 
		if (len(contact_pred_positions) == 0):
			list_zero = [0] * len(positions)
			dewet_nn_list = dewet_nn_list + list_zero
```
If there were atoms that were dewetted at a potential value, the same procedure to find the number of nearest neighbors was used - all atoms were clustered. However, in order to only count atoms that had been dewetted at a potential value, atoms in clusters had to be represented by their position so that they could be compared to a list containing the positions of atoms that have dewetted at that potential value. 

```
# iterate through clusters made to create new 'final_clusters' containing atom positions
final_clusters_positions = []
for i in range(0, len(final_clusters)):
	cluster_positions = []
	for j in range(0, len(final_clusters[i])):
		cluster = list(final_clusters[i])
		cluster_positions.append(list(positions[cluster[j]]))
	final_clusters_positions.append(cluster_positions)
```
Next, all atom positions were iterated through, and located in `final_clusters_positions`. Once its cluster was found, the atom positions in the cluster were iterated through and compared to the atom positions in `contact_pred_positions` to determine if it is dewetted.

```
for i in range(0, len(positions)):
	position = list(positions[i])
	for j in range(0, len(final_clusters_positions)):					
		if position in final_clusters_positions[j]:
			num_dewet = 0
			for k in range(0, len(final_clusters_positions[j])):
				if np.all(final_clusters_positions[j][k] in contact_pred_positions):
					num_dewet += 1

			dewet_nn_list.append(num_dewet)
			break
```
This process was completed for each potential value between 0 and 2.00, at 0.04 increments. 

### True or False Positive

Columns in the dataframe were created to indicate if an atom was a part of the PPI and if an atom was predicted to be a part of the PPI at the potential value associated with it, named 'Interface' and 'Prediction' respectively. All atoms that had were not predicted to be a part of the interface (had a value of 0 in the 'Prediction') were dropped, and if an atom was both a part of the PPI and predicted to be a part of the PPI, it would have a value of 1 in a new column labeled 'True Positive'. The model would be attempting to classify the test data as either a true positive (1) or false positive (0). 

```
df_final = df_final.drop(df_final[df_final.Prediction == 0].index)
df_final['True Positive'] = (df_final['Interface'] == 1) & (df_final['Prediction'] == 1)
df_final = df_final.drop(columns=['Prediction', 'Interface'])
```

## Model 

The model is shown below. 

```
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
	        self.fc1 = nn.Linear(62, 42) 
	        self.fc2 = nn.Linear(42, 32)
	        self.fc3 = nn.Linear(32, 16)
	        self.fc4 = nn.Linear(16, 2)

	
	def forward(self, x):
	    x = (F.selu(self.fc1(x)))
	    x = (F.selu(self.fc2(x)))
	    x = (F.selu(self.fc3(x)))
	    x = self.fc4(x)
	    return F.log_softmax(x, dim=0)


net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.009371318654565915, weight_decay=4.977688880327612e-06)
```

Elements of the model such as the activation function, learning rate, weight decay, optimization algorithm, and dropout rate were all optimized using the [HyperSearch](https://github.com/kevinzakka/hypersearch) API. 

### Training

The training of the model is shown below. 
```
criterion = nn.CrossEntropyLoss()
# train neural network
net.train()
batch_size = 10
for epoch in range(30):
	running_loss = 0.0
	total_batches = 0
	for i in range(np.shape(x_train)[0] / batch_size):
		batch = batch_size * i 
		x_batch = x_train[batch:batch + batch_size]
		y_batch = y_train[batch:batch + batch_size]

		x_batch = torch.from_numpy(x_batch).type(torch.FloatTensor)
		y_batch = torch.from_numpy(y_batch).type(torch.long)
		optimizer.zero_grad()
		net_out = net(x_batch)
		loss = criterion(net_out, y_batch)
		loss.backward()
		optimizer.step()

		running_loss += loss.data
		total_batches += 1
	print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / total_batches))
	running_loss = 0.0
	total_batches = 0
```

The loss criterion used was cross entropy loss. The model was trained for 30 epochs, with 10 atoms being passed into the model at a time; the loss value was printed after every epoch.
```
[1,   633] loss: 0.719
[2,   633] loss: 0.713
[3,   633] loss: 0.708
...
[29,   633] loss: 0.599
[30,   633] loss: 0.596
```

### Testing

The testing of the model is shown below. 
```
# test neural network

net.eval()
correct = 0
total = 0
true_positive = 0
false_positive = 0
wrong_true_positive = 0
wrong_false_positive = 0
for i in range(0, np.shape(x_test)[0]):
	x_batch, y_batch = x_test[i], y_test[i]
	x_batch = torch.from_numpy(x_batch).type(torch.FloatTensor)
	y_batch = torch.tensor([y_batch])
	outputs = net(x_batch)
	_, predicted = torch.max(outputs.data, 0)
	total += y_batch.size(0)
	if (predicted.item() == 1) & (y_batch.item() == 0):
		print('WRONG FALSE POSITIVE')
		false_positive += 1
		wrong_false_positive += 1
	elif (predicted.item() == 1) & (y_batch.item() == 1):
		print('CORRECT')
		true_positive += 1
	elif (predicted.item() == 0) & (y_batch.item() == 0):
		print('CORRECT')
		false_positive += 1
	else:
		print('WRONG TRUE POSITIVE')
		wrong_true_positive += 1
		true_positive += 1

	correct += (predicted == y_batch).item()


print('Accuracy of the network: %d%%' % (100 * correct / total))

print('Number of true positives: %d' % (true_positive))
print('Number of wrong true positives: %d' % (wrong_true_positive))
print('Number of false positives: %d' % (false_positive))
print('Number of wrong false positives: %d' % (wrong_false_positive))
```

One atom from the testing dataset was passed into the model at a time, and the number of correct and incorrect classifications were tracked in `wrong_true_positive` and `wrong_false_positive`. The overall accuracy of the network was determined by the number of correct classifications and total number of classifications made, stored in `correct` and `total` respectively. The line `outputs = net(x_batch)` stores the weights for the two classes, true and false positive; `_, predicted = torch.max(outputs.data, 0)` determines the index of the highest weight and stores it in `predicted`. If the values stored in `predicted` and `y_batch`, which holds the actual classification of an atom, are the same, we can increment the number of corect values in `correct += (predicted == y_batch).item()`.

## Results

The results of training and testing this model are shown below. 

```
Accuracy of the network: 41%
Number of true positives: 424
Number of wrong true positives: 87
Number of false positives: 694
Number of wrong false positives: 567
```
While the model had reasonable success in identifying true positives, it did a poor job of identifying false positives. One problem area is the limited information stored in the feature vector; while there was data regarding the number of nearest neighbors and number of dewetted nearest neighbors for an atom, the model did not know information _about_ those neighbors. A potential solution to this problem would be to use a convoluted neural network; much like an image, there are spatial dependencies in a protein, which a CNN would be able to take into consideration. 



