import numpy as np 
import MDAnalysis as mda 
import matplotlib.pyplot as plt
from clustering_culm import cluster_atoms_under, cluster_atoms_over, clustering_info_under, clustering_info_over
from clustering import cluster_atoms
import scipy.spatial as spatial
from sklearn.metrics import auc

# modified version of filter file to handle proteins with no actual_contact.pdb file

# want to reduce number of false positives - predicting that an atom is a contact atom when it is not 

u_dewet = mda.Universe('prot_atoms_dewetting_order.pdb')
phi_star = u_dewet.atoms.tempfactors
phi_star_bins = np.linspace(0, max(phi_star), num=101)
inds = np.digitize(phi_star, phi_star_bins)
positions = u_dewet.atoms.positions
positions = positions.tolist()

# need to first determine which atoms are actual contacts from pred_contact_tp_fp file

u_400 = mda.Universe('beta_phi_400/pred_contact_tp_fp.pdb')
contact = u_400.atoms.tempfactors
contact = contact.astype(int)

# clustering the atoms that are predicted to be a part of the interface
def cluster_pred_interface_atoms(beta_phi):
	filename = 'beta_phi_%03d/pred_contact_mask.dat' % (beta_phi)
	contact_pred = np.loadtxt(filename)
	contact_pred_ind = np.where(contact_pred == 1)[0].tolist()
	if len(contact_pred_ind) == 0:
		return 0

	atom_positions = [positions[i] for i in contact_pred_ind]
	bin_tree = spatial.cKDTree(atom_positions)
	distances, neighbors = bin_tree.query(atom_positions, k=len(atom_positions), distance_upper_bound=4)
	if len(neighbors) == 1:
		return 0

	# create list of sets
	clusters = []
	for l in neighbors:
		l = set(l)
		if len(atom_positions) in l:
			l.remove(len(atom_positions))
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

	return final_clusters

# function to cluster a list of atom indices (findings clusters among FP and TP)
def cluster_atom_ind(list_ind):
	if len(list_ind) == 0:
		return 0, 0, 0, []

	atom_positions = [positions[i] for i in list_ind]
	bin_tree = spatial.cKDTree(atom_positions)
	distances, neighbors = bin_tree.query(atom_positions, k=len(atom_positions), distance_upper_bound=0)
	if len(neighbors) == 1:
		return 0, 0, 0, []

	# create list of sets
	clusters = []
	for l in neighbors:
		l = set(l)
		if len(atom_positions) in l:
			l.remove(len(atom_positions))
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

	if final_clusters == 0:
		return 0, 0, 0, []

	# get largest cluster, smallest cluster, average size of clusters 
	# iterate through all clusters in final_clusters
	min_size = len(final_clusters[0])
	max_size = len(final_clusters[0])
	sum_size = 0
	num_clusters = len(final_clusters)

	# find smallest cluster not equal to 1, 2, or 3
	for cluster in final_clusters:
		sum_size += len(cluster)
		if (len(cluster) < min_size) and (len(cluster) != 1) and (len(cluster) != 2) and (len(cluster) != 3):
			min_size = len(cluster)
		elif min_size == 1 or min_size == 2 or min_size == 3:
			min_size = len(cluster)
		elif len(cluster) > max_size:
			max_size = len(cluster)
		else: 
			continue 

	mean_size = sum_size / num_clusters

	return min_size, max_size, mean_size, final_clusters

def clustering_info_pred_interface_atoms(beta_phi):
	filename = 'beta_phi_%03d/pred_contact_mask.dat' % (beta_phi)
	contact_pred = np.loadtxt(filename)
	total_clusters = cluster_pred_interface_atoms(beta_phi)
	if total_clusters == 0:
		return 0, 0, [], [], []

	num_clusters = len(total_clusters)
	cluster_dict = {}
	singleton_ind = []
	doubles_ind = []
	triples_ind = []
	contact_pred_ind = np.where(contact_pred == 1)[0].tolist()
	atom_positions = [positions[i] for i in contact_pred_ind]
	for cluster in total_clusters:
		if num_clusters == 0:
			continue
		else: 
			if len(cluster) in cluster_dict:
				cluster_dict[len(cluster)] += 1
			else: 
				cluster_dict[len(cluster)] = 1

		if len(cluster) == 1:
			for ind in cluster: 
				point = [atom_positions[ind][0], atom_positions[ind][1], atom_positions[ind][2]]
				index = positions.index(point)
				singleton_ind.append(index)
		elif len(cluster) == 2:
			for ind in cluster:
				point = [atom_positions[ind][0], atom_positions[ind][1], atom_positions[ind][2]]
				index = positions.index(point)
				doubles_ind.append(index)
		elif len(cluster) == 3:
			for ind in cluster:
				point = [atom_positions[ind][0], atom_positions[ind][1], atom_positions[ind][2]]
				index = positions.index(point)
				triples_ind.append(index)
		else:
			continue

	return num_clusters, cluster_dict, singleton_ind, doubles_ind, triples_ind

# function to determine indices of false positives for analysis at given beta phi factor
def false_positives(beta_phi):
	filename = 'beta_phi_%03d/pred_contact_mask.dat' % (beta_phi)
	contact_pred = np.loadtxt(filename)
	contact_ind = np.where((contact == -2) | (contact == -1) | (contact == 0))
	contact_pred_ind = np.where(contact_pred == 1)
	FP = np.intersect1d(contact_ind, contact_pred_ind)

	return FP 

# function to determine indices of true positives for analysis at given beta phi factor
def true_positives(beta_phi): 
	filename = 'beta_phi_%03d/pred_contact_mask.dat' % (beta_phi)
	contact_pred = np.loadtxt(filename)
	contact_ind = np.where(contact == 1)
	contact_pred_ind = np.where(contact_pred == 1)
	TP = np.intersect1d(contact_ind, contact_pred_ind)

	return TP

# function to get dictionaries that have indices of TP and FP for each beta phi value
def get_TP_FP():
	FP_dict = {}
	TP_dict = {}
	for i in range(0, 264, 4):
		FP = false_positives(i)
		TP = true_positives(i)
		FP_dict[i] = FP
		TP_dict[i] = TP

	return FP_dict, TP_dict

# create dictionaries of FP and TP for beta phi values 
FP_dict = {}
FP_dict_len = {}
TP_dict = {}
TP_dict_len = {}
for i in range(0, 404, 4):
	FP = false_positives(i)
	TP = true_positives(i)
	FP_dict[i] = FP
	FP_dict_len[i] = len(FP)
	TP_dict[i] = TP
	TP_dict_len[i] = len(TP)

# create dictionaries for max, min, average size of clusters for each beta phi value
FP_max_clusters = {}
TP_max_clusters = {}

FP_min_clusters = {}
TP_min_clusters = {}

FP_av_clusters = {}
TP_av_clusters = {}

for i in range(0, 404, 4):
	print(i)
	FP_min_size, FP_max_size, FP_mean_size, FP_tot_clusters = cluster_atom_ind(FP_dict[i])
	TP_min_size, TP_max_size, TP_mean_size, TP_tot_clusters = cluster_atom_ind(TP_dict[i])

	FP_max_clusters[i] = FP_max_size
	TP_max_clusters[i] = TP_max_size

	FP_min_clusters[i] = FP_min_size
	TP_min_clusters[i] = TP_min_size

	FP_av_clusters[i] = FP_mean_size
	TP_av_clusters[i] = TP_mean_size

# visualize plots for FP and TP cluster information

def visualize_max():
	font = {'size': 8}
	barwidth = 0.9
	fig, ax = plt.subplots()
	FP_bars = plt.bar(FP_max_clusters.keys(), FP_max_clusters.values(), barwidth, label="False Positive")
	TP_bars = plt.bar(np.array(TP_max_clusters.keys()) + barwidth, TP_max_clusters.values(), barwidth, label="True Positive")
	plt.title("Size of Largest Cluster Among False and True Positives for Given Beta Phi Value", fontdict=font)
	plt.xlabel("Beta Phi Value")
	plt.ylabel("Size of Largest Cluster")
	plt.xlim(0, 202)
	plt.ylim(0, 60)
	plt.xticks(np.arange(0, 204, 4.0))
	plt.tick_params(labelsize=6)
	plt.legend()
	plt.tight_layout()
	plt.show()

def visualize_min():
	font = {'size': 8}
	barwidth = 0.9
	fig, ax = plt.subplots()
	FP_bars = plt.bar(FP_min_clusters.keys(), FP_min_clusters.values(), barwidth, label="False Positive")
	TP_bars = plt.bar(np.array(TP_min_clusters.keys()) + barwidth, TP_min_clusters.values(), barwidth, label="True Positive")
	plt.title("Size of Smallest Cluster Among False and True Positives for Given Beta Phi Value", fontdict=font)
	plt.xlabel("Beta Phi Value")
	plt.ylabel("Size of Smallest Cluster")
	plt.xlim(200, 402)
	plt.ylim(0, 15)
	plt.xticks(np.arange(200, 404, 4.0))
	plt.tick_params(labelsize=6)
	plt.legend()
	plt.tight_layout()
	plt.show()

def visualize_mean():
	font = {'size': 8}
	barwidth = 0.9
	fig, ax = plt.subplots()
	FP_bars = plt.bar(FP_av_clusters.keys(), FP_av_clusters.values(), barwidth, label="False Positive")
	TP_bars = plt.bar(np.array(TP_av_clusters.keys()) + barwidth, TP_av_clusters.values(), barwidth, label="True Positive")
	plt.title("Mean Cluster Size Among False and True Positives for Given Beta Phi Value", fontdict=font)
	plt.xlabel("Beta Phi Value")
	plt.ylabel("Size of Mean Cluster")
	plt.xlim(200, 402)
	plt.xticks(np.arange(200, 404, 4.0))
	plt.tick_params(labelsize=6)
	plt.legend()
	plt.tight_layout()
	plt.show()

# for each beta phi factor determine indices of singleton clusters and remove intersection between singleton and FP, TP

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



# create dictionary that gives indices of TP and FP removed
FP_removed = {}
TP_removed = {}

for i in range(0, 404, 4):
	prefilter_FP = FP_dict[i]
	postfilter_FP = FP_dict_filter[i]
	difference_FP = list(set(prefilter_FP) - set(postfilter_FP))
	FP_removed[i] = difference_FP

	prefilter_TP = TP_dict[i]
	postfilter_TP = TP_dict_filter[i]
	difference_TP = list(set(prefilter_TP) - set(postfilter_TP))
	TP_removed[i] = difference_TP

def visualize_FP_filter():
	font = {'size': 10}
	plt.bar(list(FP_removed.keys()), FP_removed.values())
	plt.xlabel("Beta Phi Value")
	plt.ylabel("Number of False Positives Removed")
	plt.xlim(200, 400)
	plt.xticks(np.arange(200, 402, 4.0))
	plt.tick_params(labelsize=6)
	plt.title(r'Number of False Postives Removed, Filtering out Clusters of Size 1, 2 and 3', fontdict=font)
	plt.show()

def visualize_TP_filter():
	font = {'size': 8}
	plt.bar(list(TP_removed.keys()), TP_removed.values())
	plt.xlabel("Beta Phi Value")
	plt.ylabel("Number of True Positives Removed")
	plt.xlim(200, 400)
	plt.ylim(0, 14)
	plt.xticks(np.arange(200, 402, 4.0))
	plt.tick_params(labelsize=6)
	plt.title(r'Number of True Postives Removed, Filtering out Clusters of Size 1, 2 and 3', fontdict=font)
	plt.show()

def roc():
	P = len(contact[contact == 1])
	font = {'size': 8}
	FPR_pre = [len(x) / float(P) for x in FP_dict.values()]
	TPR_pre = [len(x) / float(P) for x in TP_dict.values()]
	FPR_post = [len(x) / float(P) for x in FP_dict_filter.values()]
	TPR_post = [len(x) / float(P) for x in TP_dict_filter.values()]
	indices_pre = np.argsort(FPR_pre)
	indices_post = np.argsort(FPR_post)
	FPR_pre_ordered = (np.asarray(FPR_pre))[indices_pre]
	TPR_pre_ordered = (np.asarray(TPR_pre))[indices_pre]
	FPR_post_ordered = (np.asarray(FPR_post))[indices_post]
	TPR_post_ordered = (np.asarray(TPR_post))[indices_post]

	plt.plot(FPR_post, TPR_post, label='Post-Filter')
	plt.plot(FPR_pre, TPR_pre, label='Pre-Filter')
	pre_filter_area = auc(FPR_pre_ordered, TPR_pre_ordered)
	post_filter_area = auc(FPR_post_ordered, TPR_post_ordered)
	print('Area Under Pre-Filter ROC Curve: %f' % (pre_filter_area))
	print('Area Under Post-Filter ROC Curve: %f' % (post_filter_area))
	plt.text(0.22, 0.016, 'Area Under Pre-Filter ROC Curve: %f' % (pre_filter_area), fontsize=9)
	plt.text(0.22, 0.01, 'Area Under Post-Filter ROC Curve: %f' % (post_filter_area), fontsize=9)
	plt.title('ROC Curve, Pre-Filter and Post-Filter')
	plt.legend()
	plt.show()

def visualize_pre_FPR_TPR():
	P = len(contact[contact == 1])
	font = {'size': 8}
	FPR_pre = [len(x) / float(P) for x in FP_dict.values()]
	TPR_pre = [len(x) / float(P) for x in TP_dict.values()]
	indices_pre = np.argsort(FPR_pre)
	FPR_pre_ordered = (np.asarray(FPR_pre))[indices_pre]
	TPR_pre_ordered = (np.asarray(TPR_pre))[indices_pre]
	
	plt.plot(FP_dict.keys(), FPR_pre_ordered, label='FPR')
	plt.plot(FP_dict.keys(), TPR_pre_ordered, label='TPR')
	plt.title('FPR and TPR, Pre-Filter')
	plt.legend()
	plt.show()

#visualize_pre_FPR_TPR()

def visualize_post_FPR_TPR():
	P = len(contact[contact == 1])
	font = {'size': 8}
	FPR_post = [len(x) / float(P) for x in FP_dict_filter.values()]
	TPR_post = [len(x) / float(P) for x in TP_dict_filter.values()]
	indices_post = np.argsort(FPR_post)
	FPR_post_ordered = (np.asarray(FPR_post))[indices_post]
	TPR_post_ordered = (np.asarray(TPR_post))[indices_post]

	plt.plot(FP_dict.keys(), FPR_post_ordered, label='FPR')
	plt.plot(FP_dict.keys(), TPR_post_ordered, label='TPR')
	plt.title('FPR and TPR, Post-Filter')
	plt.legend()
	plt.show()

#visualize_post_FPR_TPR()

# create atom groups for visualizations

contact_ind = np.where(contact == 1)[0].tolist()
ag_contact = mda.AtomGroup(contact_ind, u_dewet)

ag_filtered_FP = []
ag_filtered_TP = []
ag_FP = []
ag_TP = []
for i in range(0, 404, 4):
	ag = mda.AtomGroup(FP_dict[i], u_dewet)
	ag_FP.append(ag)

	ag = mda.AtomGroup(FP_removed[i], u_dewet)
	ag_filtered_FP.append(ag)

	ag = mda.AtomGroup(TP_dict[i], u_dewet)
	ag_TP.append(ag)

	ag = mda.AtomGroup(TP_removed[i], u_dewet)
	ag_filtered_TP.append(ag)

# vmd visualizations to visualize filtering

with mda.selections.vmd.SelectionWriter('filtering.vmd', mode='w') as vmd:
	# visualization for actual contacts
	vmd.write(ag_contact, number=None, name='contacts', frame=None)	

	# visualization for false positives
	beta_phi = 0
	for ag in ag_FP:
		if len(ag) == 0:
			beta_phi += 4
		else: 
			vmd.write(ag, number=None, name='FP.%d' % (beta_phi), frame=None)
			beta_phi += 4

	# visualization for true positives
	beta_phi = 0
	for ag in ag_TP:
		if len(ag) == 0:
			beta_phi += 4
		else: 
			vmd.write(ag, number=None, name='TP.%d' % (beta_phi), frame=None)
			beta_phi += 4

	# visualization for false positives that are filtered out
	beta_phi = 0
	for ag in ag_filtered_FP:
		if len(ag) == 0:
			beta_phi += 4
		else: 
			vmd.write(ag, number=None, name='FP.filter.%d' % (beta_phi), frame=None)
			beta_phi += 4

	# visualization for true positives that are filtered out
	beta_phi = 0
	for ag in ag_filtered_TP:
		if len(ag) == 0:
			beta_phi += 4
		else: 
			vmd.write(ag, number=None, name='TP.filter.%d' % (beta_phi), frame=None)
			beta_phi += 4

