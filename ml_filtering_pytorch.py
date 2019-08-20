import numpy as np 
from numpy import diff
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from scipy.signal import argrelextrema
import scipy.spatial as spatial
import pandas as pd
from biopandas.pdb import PandasPdb 
import MDAnalysis as mda 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# proteins to train and test model 
protein_files = ['MDM2', 'Phospholipase_a2', 'mannose_binding_protein', 'ubiquitin', 'thymidylate_synthase']

# function that will return a list containing the number of nearest neighbors for atoms in a given protein

def get_num_nn(protein_name):
	u = mda.Universe('%s/prot_atoms_dewetting_order.pdb' % (protein_name))
	positions = u.atoms.positions
	nn_tree = spatial.cKDTree(positions)
	distances, neighbors = nn_tree.query(positions, k=len(positions), distance_upper_bound=1.5)

	if len(neighbors) == 1:
		return 0

	# create list of sets
	clusters = []
	for l in neighbors:
		l = set(l)
		if len(positions) in l:
			l.remove(len(positions))
		clusters.append(l)

	# traverse the nearest neighbors of each atom in each bin to determine clusters

	final_clusters = []
	ind_remove = []
	loop_bool = True
	for i in range(len(clusters)):
		loop_bool = True
		while loop_bool:
			if i in ind_remove:
				i += 1
			else: 
				loop_bool = False
		if i == len(clusters):
			break

		cluster = clusters[i]

		j = 0
		while j <= len(clusters) - 1:
			if len(cluster & (clusters[j])) == 0:
				j += 1
			elif j in ind_remove:
				j += 1
			elif i == j:
				j += 1
			else:
				cluster = cluster.union(clusters[j])
				ind_remove.append(j)
				ind_remove.append(i)
				j = 0 
		
		if cluster in final_clusters:
			continue
		else: 
			final_clusters.append(cluster)

	nn_list = []
	for i in range(0, len(positions)):
		for j in range(0, len(final_clusters)):
			if i in final_clusters[j]:
				nn_list.append(len(final_clusters[j]))
				break

	nn_list = nn_list * 51

	return nn_list

# function that will return a list containing the number of nearest dewetted neighbors for atoms in a given protein

def get_num_dewet_nn(protein_name):
	dewet_nn_list = []
	u = mda.Universe('%s/prot_atoms_dewetting_order.pdb' % (protein_name))
	tempfactors = u.atoms.tempfactors
	positions = u.atoms.positions
	for i in range(0, 204, 4): 
		filename = protein_name + ('/beta_phi_%03d/pred_contact_mask.dat' % (i))
		contact_pred = np.loadtxt(filename)
		contact_pred_positions = positions[np.where(contact_pred == 1)[0].tolist()] 
		if (len(contact_pred_positions) == 0):
			list_zero = [0] * len(positions)
			dewet_nn_list = dewet_nn_list + list_zero
		else: 

			tree = spatial.cKDTree(contact_pred_positions)
			distances, neighbors = tree.query(contact_pred_positions, k=len(contact_pred_positions), distance_upper_bound=1.5)

			if len(neighbors) == 1:
				list_zero = [0] * len(positions)
				dewet_nn_list = dewet_nn_list + list_zero
			else: 
				# create list of sets
				clusters = []
				for l in neighbors:
					l = set(l)
					if len(contact_pred_positions) in l:
						l.remove(len(contact_pred_positions))
					clusters.append(l)

				# traverse the nearest neighbors of each atom in each bin to determine clusters

				final_clusters = []
				ind_remove = []
				loop_bool = True
				for i in range(len(clusters)):
					loop_bool = True
					while loop_bool:
						if i in ind_remove:
							i += 1
						else: 
							loop_bool = False
					if i == len(clusters):
						break

					cluster = clusters[i]

					j = 0
					while j <= len(clusters) - 1:
						if len(cluster & (clusters[j])) == 0:
							j += 1
						elif j in ind_remove:
							j += 1
						elif i == j:
							j += 1
						else:
							cluster = cluster.union(clusters[j])
							ind_remove.append(j)
							ind_remove.append(i)
							j = 0 
					
					if cluster in final_clusters:
						continue
					else: 
						final_clusters.append(cluster)

				# iterate through clusters made to create new 'final_clusters' containing atom positions
				final_clusters_positions = []
				for i in range(0, len(final_clusters)):
					cluster_positions = []
					for j in range(0, len(final_clusters[i])):
						cluster = list(final_clusters[i])
						cluster_positions.append(contact_pred_positions[cluster[j]])
					final_clusters_positions.append(cluster_positions)
				

				for i in range(0, len(contact_pred)):
					if contact_pred[i] == 0:
						dewet_nn_list.append(0)
					else:
						position = positions[i]
						for j in range(0, len(final_clusters_positions)):
							if np.all(position == final_clusters_positions[j][0]):
								dewet_nn_list.append(len(final_clusters_positions[j]))
								break

							if j == (len(final_clusters_positions) - 1):
								dewet_nn_list.append(0)

	return dewet_nn_list


def get_data():

	first = True
	for protein in protein_files:
		filename = '%s/prot_atoms_dewetting_order.pdb' % (protein)
		ppdb = PandasPdb()
		ppdb.read_pdb(filename)

		if protein == 'thymidylate_synthase':
			u = mda.Universe('%s/actual_contact.pdb' % (protein))
			contact = u.atoms.tempfactors
			contact_ind = np.where(contact == 1)[0].tolist()
			ag_contact = mda.AtomGroup(contact_ind, u)
			interface_col = np.where(contact == 1, 1, 0)
		else:
			u = mda.Universe(protein + '/beta_phi_400/pred_contact_tp_fp.pdb')
			contact = u.atoms.tempfactors.astype(int)
			interface_col = np.where(contact == 1, 1, 0)

		# create Pandas dataframe
		df = ppdb.df['ATOM']
		df = df.drop(columns=['record_name', 'atom_number', 'residue_number', 'x_coord', 'y_coord', 'z_coord', 'chain_id', 'blank_1', 'alt_loc', 'blank_2', 'insertion', 'blank_3', 'occupancy', 'blank_4', 
							  'segment_id', 'element_symbol', 'charge', 'line_idx'])

		# encode nominal variables
		df = pd.get_dummies(df, columns=['atom_name'], prefix=['atom_name'])
		df = pd.get_dummies(df, columns=['residue_name'], prefix=['residue_name'])

		# add column that indicates whether atom is part of interface 
		df['Interface'] = interface_col

		# for each beta phi value, access atoms that are predicted to be a part of the interface and compare to 
		# the actual contact atoms
		if protein == 'thymidylate_synthase':
			start = 0
			end = 204
		else:
			start = 0
			end = 204
		for i in range(start, end, 4):
			print(i)
			df_concat = df
			df_concat['Beta Phi Prediction Value'] = float(i) / 100
			filename = protein + '/beta_phi_%03d/pred_contact_mask.dat' % (i)
			contact_pred = np.loadtxt(filename)
			df_concat['Prediction'] = contact_pred

			# df_concat = df_concat.drop(df_concat[df_concat.Prediction == 0].index)
			# df_concat['True Positive'] = (df_concat['Interface'] == 1) & (df_concat['Prediction'] == 1)
			# df_concat = df_concat.drop(columns=['Prediction', 'Interface'])

			if i == 0:
				df_final = df_concat
			else:
				df_final = pd.concat([df_final, df_concat], ignore_index=True)

		nn_col = get_num_nn(protein)
		dewet_nn_col = get_num_dewet_nn(protein)

		df_final['Number Nearest Neighbors'] = nn_col
		df_final['Number Dewetted Nearest Neighbors'] = dewet_nn_col

		df_final = df_final.drop(df_final[df_final.Prediction == 0].index)
		df_final['True Positive'] = (df_final['Interface'] == 1) & (df_final['Prediction'] == 1)
		df_final = df_final.drop(columns=['Prediction', 'Interface'])

		if first:
			df_final_final = df_final
			first = False
		else:
			df_final_final = pd.concat([df_final_final, df_final], ignore_index=True)


	df_final_final[['True Positive']] *= 1
	df_final_final = df_final_final.loc[df_final_final['b_factor'] > -2]

	# randomize rows in dataframe 
	df_final_final = df_final_final.sample(frac=1)
	df_final_final = df_final_final.fillna(0)

	x = df_final_final.iloc[:, df_final_final.columns != 'True Positive'].values
	y = df_final_final['True Positive'].values

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

	# scale values
	sc = StandardScaler()
	x_train = sc.fit_transform(x_train)
	x_test = sc.transform(x_test)

	return x_train, x_test, y_train, y_test, df_final_final

x_train, x_test, y_train, y_test, df_train = get_data()

# scale values
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test) 

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
	        self.fc1 = nn.Linear(62, 42) 
	        self.fc2 = nn.Linear(42, 32)
	        self.fc3 = nn.Linear(32, 16)
	        #self.dropout = nn.Dropout(p=0.384644071093)
	        self.fc4 = nn.Linear(16, 2)

	
	def forward(self, x):
	    x = (F.relu(self.fc1(x)))
	    x = (F.relu(self.fc2(x)))
	    x = (F.relu(self.fc3(x)))
	    x = self.fc4(x)
	    return F.log_softmax(x, dim=0)


net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.009280582900906153, weight_decay=9.791321988936709e-06)
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


print('Done')

# test neural network]

net.eval()
correct = 0
total = 0
wrong_true_positive = 0
true_positive = 0
false_positive = 0
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

