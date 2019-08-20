import MDAnalysis as mda 
import numpy as np 
from filter import cluster_pred_interface_atoms, cluster_atom_ind, get_TP_FP, clustering_info_pred_interface_atoms, visualize_post_FPR_TPR
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from sklearn.metrics import auc
from collections import OrderedDict

# change directory to the protein in question

u_dewet = mda.Universe('prot_atoms_dewetting_order.pdb')
phi_star = u_dewet.atoms.tempfactors
phi_star_bins = np.linspace(0, max(phi_star), num=101)
inds = np.digitize(phi_star, phi_star_bins)
positions = u_dewet.atoms.positions
positions = positions.tolist()

# beta factor = 1, then it is an actual contact atom, beta factor = -2, not a contact atom
u = mda.Universe('actual_contact.pdb')
contact = u.atoms.tempfactors

# adaptive filter: iterate through clusters, create dictionary with key as cluster size and value as indices of atoms that 
# are a part of cluster of a given size

# for each beta phi value, determine size of smallest true positive cluster, and filter out all clusters that have a size
# smaller than smallest true positive cluster

# function to get dictionary with key as cluster size and value as indices of atoms that 
# are a part of cluster of a given size
def clustering_info_adaptive_filter(beta_phi):
	filename = 'beta_phi_%03d/pred_contact_mask.dat' % (beta_phi)
	contact_pred = np.loadtxt(filename)
	total_clusters = cluster_pred_interface_atoms(beta_phi)

	if total_clusters == 0:
		return 0, 0

	num_clusters = len(total_clusters)
	cluster_dict = {}
	contact_pred_ind = np.where(contact_pred == 1)[0].tolist()
	atom_positions = [positions[i] for i in contact_pred_ind]

	for cluster in total_clusters:
		if num_clusters == 0:
			return 0, 0
		else:
			if len(cluster) in cluster_dict:
				for ind in cluster:
					point = [atom_positions[ind][0], atom_positions[ind][1], atom_positions[ind][2]]
					index = positions.index(point)
					cluster_dict[len(cluster)].append(index)
			else:
				cluster_dict[len(cluster)] = []
				for ind in cluster:
					point = [atom_positions[ind][0], atom_positions[ind][1], atom_positions[ind][2]]
					index = positions.index(point)
					cluster_dict[len(cluster)].append(index)

	return num_clusters, cluster_dict


# for each beta phi factor determine size of smallest true positive cluster, and filter out all clusters that have a size
# smaller than smallest true positive cluster

# FP_dict, TP_dict = get_TP_FP()
# FP_dict_filter = {}
# TP_dict_filter = {}

# for i in range(0, 404, 4):
# 	print(i)	
# 	min_size, max_size, mean_size, total_clusters = cluster_atom_ind(TP_dict[i])
# 	num_clusters, cluster_dict = clustering_info_adaptive_filter(i)
# 	print(cluster_dict)
# 	print(min_size)
# 	if cluster_dict == 0:
# 		FP_dict_filter[i] = FP_dict
# 		TP_dict_filter[i] = TP_dict
# 	else:
# 		if min_size == 0:
# 			new_ind_FP = FP_dict
# 			new_ind_TP = TP_dict
# 		elif min_size == 1:
# 			if 1 in cluster_dict:
# 				new_ind_FP = list(set(FP_dict[i]) - set(cluster_dict[1]))
# 				new_ind_TP = list(set(TP_dict[i]) - set(cluster_dict[1]))
# 			else:
# 				new_ind_FP = FP_dict
# 				new_ind_TP = TP_dict
# 		else:
# 			j = 1
# 			new_ind_FP = FP_dict[i]
# 			new_ind_TP = TP_dict[i]
# 			while j < min_size:
# 				if j in cluster_dict:
# 					new_ind_FP = list(set(new_ind_FP) - set(cluster_dict[j]))
# 					new_ind_TP = list(set(new_ind_TP) - set(cluster_dict[j]))
# 					j += 1
# 				else:
# 					j += 1

# 		FP_dict_filter[i] = new_ind_FP
# 		TP_dict_filter[i] = new_ind_TP

# function that calculates atom positions that should be filtered out for single beta phi value 

def adaptive_filtering(beta_phi):
	FP_dict, TP_dict = get_TP_FP()
	min_size, max_size, mean_size, total_clusters = cluster_atom_ind(TP_dict[beta_phi])
	num_clusters, cluster_dict = clustering_info_adaptive_filter(beta_phi)

	if cluster_dict == 0:
		return [], []
	else: 
		if min_size == 0:
			return [], []
		elif min_size == 1:
			if 1 in cluster_dict:
				return cluster_dict[1]
			else:
				return [], []

		else: 
			FP_remove = []
			TP_remove = []
			j = 1
			while j < min_size:
				if j in cluster_dict:
					for pos in cluster_dict[j]:
						if pos in FP_dict[beta_phi]:
							FP_remove.append(pos)
						else: 
							TP_remove.append(pos)
					j += 1
				else:
					j += 1

			return FP_remove, TP_remove

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

# filtering clusters smaller than smallest true positives cluster was not very effective

# alternative way to filter: for low beta phi values, determining positions of atoms that are labeled
# as false positives; for subsequent beta phi values, filtering out atoms that are in the vicinity of atoms
# that were previously labeled to be false positives

# need to have a high confidence that atoms that are labeled as false positives are actually false positives

# this algorithm uses fact that we know what the false positives and true positives are - idea is to create algorithm
# that can detect false positives with high confidence without knowing the truth 

# as of now, there really is no way to distinguish false positive and true positive clusters - deciding to not care about 
# which is which 

filtered_ind = []

def recursive_filtering(beta_phi):
	num_clusters, cluster_dict, singleton_ind, doubles_ind, triples_ind = clustering_info_pred_interface_atoms(beta_phi)
	if len(singleton_ind) == 0 and len(doubles_ind) == 0 and len(triples_ind) == 0:
		return list(set(filtered_ind))
	else:
		
		if len(filtered_ind) == 0:
			filtered_ind.extend(singleton_ind)
			filtered_ind.extend(doubles_ind)
			filtered_ind.extend(triples_ind)
		else:

			filename = 'beta_phi_%03d/pred_contact_mask.dat' % (beta_phi)
			contact_pred = np.loadtxt(filename)
			contact_pred_ind = np.where(contact_pred == 1)[0].tolist()
			contact_pred_ind.extend(filtered_ind)
			total_ind = set(contact_pred_ind)
			total_positions = [positions[i] for i in total_ind]

			min_size, max_size, mean_size, total_clusters = cluster_atom_ind(total_ind)

			# determine the clusters that have atoms included in filtered_ind - based on position values 

			filtered_ind_pos = [positions[i] for i in filtered_ind]
			for cluster in total_clusters:
				cluster = list(set(cluster))
				cluster_pos = [total_positions[i] for i in cluster]
				intersection = [x for x in cluster_pos if x in filtered_ind_pos]
				if len(intersection) != 0:
					for position in cluster_pos:
						filtered_ind.append(positions.index(position))
				else:
					continue

			# option to add singleton, doubles, and triples clusters into filtered_ind as well - compare ROC for both

			filtered_ind.extend(singleton_ind)
			# filtered_ind.extend(doubles_ind)
			# filtered_ind.extend(triples_ind)

	print(list(set(filtered_ind)))
	return list(set(filtered_ind))

FP_dict, TP_dict = get_TP_FP()
# FP_dict_filter = {}
# TP_dict_filter = {}

# for i in range(0, 404, 4):
# 	print(i)
# 	ind_to_filter = recursive_filtering(i)
# 	if len(ind_to_filter) == 0:
# 		FP_dict_filter[i] = FP_dict[i]
# 		TP_dict_filter[i] = TP_dict[i]
# 	else:
# 		# for singleton clusters
# 		new_ind_FP = list(set(FP_dict[i]) - set(ind_to_filter))
# 		new_ind_TP = list(set(TP_dict[i]) - set(ind_to_filter))

# 		FP_dict_filter[i] = new_ind_FP
# 		TP_dict_filter[i] = new_ind_TP

# roc()
# visualize_post_FPR_TPR()

# for every beta phi value calculate area under the curve - adaptive filtering

P = len(contact[contact == 1])
FPR_pre = [len(x) / float(P) for x in FP_dict.values()]
TPR_pre = [len(x) / float(P) for x in TP_dict.values()]
indices_pre = np.argsort(FPR_pre)
FPR_pre_ordered = (np.asarray(FPR_pre))[indices_pre]
TPR_pre_ordered = (np.asarray(TPR_pre))[indices_pre]

pre_filter_area = auc(FPR_pre_ordered, TPR_pre_ordered)

roc_area = OrderedDict()

# for i in range(0, 264, 4):
# 	print(i)
# 	FP_dict_filter = {}
# 	TP_dict_filter = {}
# 	ind_to_filter = recursive_filtering(i)
# 	# ag = mda.AtomGroup(ind_to_filter, u_dewet)
# 	# ag_list.append(ag)
# 	for j in range(0, 264, 4):
# 		if len(ind_to_filter) == 0:
# 			break
# 		else:
# 			new_ind_FP = list(set(FP_dict[j]) - set(ind_to_filter))
# 			new_ind_TP = list(set(TP_dict[j]) - set(ind_to_filter))
# 			FP_dict_filter[j] = new_ind_FP
# 			TP_dict_filter[j] = new_ind_TP

# 	if len(ind_to_filter) == 0:
# 		roc_area[i] = pre_filter_area
# 	else:
# 		FPR_post = [len(x) / float(P) for x in FP_dict_filter.values()]
# 		TPR_post = [len(x) / float(P) for x in TP_dict_filter.values()]
# 		indices_post = np.argsort(FPR_post)
# 		FPR_post_ordered = (np.asarray(FPR_post))[indices_post]
# 		TPR_post_ordered = (np.asarray(TPR_post))[indices_post]

# 		post_filter_area = auc(FPR_post_ordered, TPR_post_ordered)
# 		roc_area[i] = post_filter_area

# for every beta phi value calculate area under the curve - cluster filtering
# also create atom group that will visualize the atoms that are being filtered out 
ag_filtered = []
for i in range(0, 264, 4):
	print(i)
	FP_dict_filter = {}
	TP_dict_filter = {}
	num_clusters, cluster_dict, singleton_ind, doubles_ind, triples_ind = clustering_info_pred_interface_atoms(i)

	singleton_ind.extend(doubles_ind)
	# singleton_ind.extend(triples_ind)
	total_ind = singleton_ind
	filtered_ag = mda.AtomGroup(total_ind, u_dewet)
	ag_filtered.append(filtered_ag)

	for j in range(0, 264, 4):
		if len(total_ind) == 0:
			break
		else:
			new_ind_FP = list(set(FP_dict[j]) - set(total_ind))
			new_ind_TP = list(set(TP_dict[j]) - set(total_ind))
			FP_dict_filter[j] = new_ind_FP
			TP_dict_filter[j] = new_ind_TP

	if len(total_ind) == 0:
		roc_area[i] = pre_filter_area
	else:
		FPR_post = [len(x) / float(P) for x in FP_dict_filter.values()]
		TPR_post = [len(x) / float(P) for x in TP_dict_filter.values()]
		indices_post = np.argsort(FPR_post)
		FPR_post_ordered = (np.asarray(FPR_post))[indices_post]
		TPR_post_ordered = (np.asarray(TPR_post))[indices_post]

		post_filter_area = auc(FPR_post_ordered, TPR_post_ordered)
		roc_area[i] = post_filter_area

# determine key that gives max value for area under curve
key = max(roc_area, key=roc_area.get)
num_clusters, cluster_dict, singleton_ind, doubles_ind, triples_ind = clustering_info_pred_interface_atoms(key)
print(key)

# ignore clusters with 3
FP_dict_filter = {}
TP_dict_filter = {}

# create list of atom groups for FP and TP filtered out
ag_FP_filtered = []
ag_TP_filtered = []

singleton_ind.extend(doubles_ind)
total_ind = singleton_ind
for j in range(0, 264, 4):
	new_ind_FP = list(set(FP_dict[j]) - set(total_ind))
	new_ind_TP = list(set(TP_dict[j]) - set(total_ind))
	FP_dict_filter[j] = new_ind_FP
	FP_removed_ag = mda.AtomGroup(list(set(FP_dict[j]) - set(FP_dict_filter[j])), u_dewet)
	ag_FP_filtered.append(FP_removed_ag)
	TP_dict_filter[j] = new_ind_TP
	TP_removed_ag = mda.AtomGroup(list(set(TP_dict[j]) - set(TP_dict_filter[j])), u_dewet)
	ag_TP_filtered.append(TP_removed_ag)

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

	plt.plot(FPR_post_ordered, TPR_post_ordered, label='Post-Filter')
	plt.plot(FPR_pre_ordered, TPR_pre_ordered, label='Pre-Filter')
	pre_filter_area = auc(FPR_pre_ordered, TPR_pre_ordered)
	post_filter_area = auc(FPR_post_ordered, TPR_post_ordered)
	print('Area Under Pre-Filter ROC Curve: %f' % (pre_filter_area))
	print('Area Under Post-Filter ROC Curve: %f' % (post_filter_area))
	plt.text(0.12, 0.016, 'Area Under Pre-Filter ROC Curve: %f' % (pre_filter_area), fontsize=9)
	plt.text(0.12, 0.01, 'Area Under Post-Filter ROC Curve: %f' % (post_filter_area), fontsize=9)
	plt.title('ROC Curve, Pre-Filter and Post-Filter')
	plt.legend()
	plt.show()

#roc()

# print(roc_area)

def visualize_roc_area():
	plt.title('Area Under ROC Curve for Beta Phi Values for Thymidylate Synthase')
	plt.plot(roc_area.keys(), roc_area.values())
	plt.text(10, 1.62, 'Optimal Beta Phi Value: %f' % (float(key) / 100), fontsize=9)
	plt.show()

visualize_roc_area()

# at the best beta phi value for filtering, use algorithm that finds atoms that are near the optimal 
# list of atoms to filter out

# best clustering distance - 1.4 A
 
def recursive_filtering_for_betaphi(key):
	key_recursive_dict = {}
	num_clusters, cluster_dict, singleton_ind, doubles_ind, triples_ind = clustering_info_pred_interface_atoms(key)
	singleton_ind.extend(doubles_ind)
	initial_ind = singleton_ind

	filtered_ind = initial_ind
	for k in range(0, 264, 4):
		print(k)
		if k <= key:
			key_recursive_dict[k] = filtered_ind
		else:
			filename = 'beta_phi_%03d/pred_contact_mask.dat' % (k)
			contact_pred = np.loadtxt(filename)
			contact_pred_ind = np.where(contact_pred == 1)[0].tolist()
			contact_pred_ind.extend(filtered_ind)
			total_ind = set(contact_pred_ind)
			total_positions = [positions[j] for j in total_ind]

			min_size, max_size, mean_size, total_clusters = cluster_atom_ind(total_ind)

			# determine the clusters that have atoms included in filtered_ind - based on position values 

			filtered_ind_pos = [positions[j] for j in filtered_ind]
			for cluster in total_clusters:
				cluster = list(set(cluster))
				cluster_pos = [total_positions[i] for i in cluster]
				intersection = [x for x in cluster_pos if x in filtered_ind_pos]
				if len(intersection) != 0:
					for position in cluster_pos:
						filtered_ind.append(positions.index(position))
				else:
					continue

			filtered_ind.extend(initial_ind)
			filtered_ind = list(set(filtered_ind))
			key_recursive_dict[k] = filtered_ind

	return key_recursive_dict

inds_to_filter = recursive_filtering_for_betaphi(key)
FP_dict_filter = {}
TP_dict_filter = {}
for i in range(0, 264, 4):
	print(i)
	if len(inds_to_filter[i]) == 0:
		continue
	else:
		new_ind_FP = list(set(FP_dict[i]) - set(inds_to_filter[i]))
		new_ind_TP = list(set(TP_dict[i]) - set(inds_to_filter[i]))
		FP_dict_filter[i] = new_ind_FP
		TP_dict_filter[i] = new_ind_TP


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

	plt.plot(FPR_post_ordered, TPR_post_ordered, label='Post-Filter')
	plt.plot(FPR_pre_ordered, TPR_pre_ordered, label='Pre-Filter')
	pre_filter_area = auc(FPR_pre_ordered, TPR_pre_ordered)
	post_filter_area = auc(FPR_post_ordered, TPR_post_ordered)
	print('Area Under Pre-Filter ROC Curve: %f' % (pre_filter_area))
	print('Area Under Post-Filter ROC Curve: %f' % (post_filter_area))
	plt.text(0.12, 0.016, 'Area Under Pre-Filter ROC Curve: %f' % (pre_filter_area), fontsize=9)
	plt.text(0.12, 0.01, 'Area Under Post-Filter ROC Curve: %f' % (post_filter_area), fontsize=9)
	plt.title('ROC Curve, Pre-Filter and Post-Filter')
	plt.legend()
	plt.show()

#roc()


# for each beta phi value, want to visualize what is being filtered out
# also visualize the atoms that would be filtered out for each beta phi value

with mda.selections.vmd.SelectionWriter('adapt_filtering.vmd', mode='w') as vmd:
	beta_phi = 0
	for ag in ag_FP_filtered:
		if len(ag) == 0:
			beta_phi += 4
		else: 
			vmd.write(ag, number=None, name='FP.ideal.filter.%d' % (beta_phi), frame=None)
			beta_phi += 4

	beta_phi = 0
	for ag in ag_TP_filtered:
		if len(ag) == 0:
			beta_phi += 4
		else: 
			vmd.write(ag, number=None, name='TP.ideal.filter.%d' % (beta_phi), frame=None)
			beta_phi += 4

	beta_phi = 0
	for ag in ag_filtered: 
		if len(ag) == 0:
			beta_phi += 4
		else:
			vmd.write(ag, number=None, name='filter.%d' % (beta_phi), frame=None)
			beta_phi += 4
