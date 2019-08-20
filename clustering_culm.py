import MDAnalysis as mda 
import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt

u = mda.Universe('prot_atoms_dewetting_order.pdb')

# list of phi_star values of heavy atoms 
phi_star = u.atoms.tempfactors

# list of positions of heavy atoms
positions = u.atoms.positions

# determine bin edges
phi_star_bins = np.linspace(0, max(phi_star), num=101)

# histogram of phi_star values
inds = np.digitize(phi_star, phi_star_bins)
hist, bin_edges = np.histogram(phi_star, bins=phi_star_bins)

positions = positions.tolist()
# function to cluster all atoms under a specified beta phi factor
def cluster_atoms_under(bin_int):

	# create list of positions of all atoms under specified beta phi factor
	bin_ind = np.where((inds < bin_int) & (phi_star > 0))[0].tolist()
	if len(bin_ind) == 0:
		return 0

	bin_positions = [positions[i] for i in bin_ind]

	# create KDTree and determine neighbors of each point in the bin
	bin_tree = spatial.cKDTree(bin_positions)
	distances, neighbors = bin_tree.query(bin_positions, k=len(bin_positions), distance_upper_bound=4)
	if len(neighbors) == 1:
		return 0

	# create list of sets
	clusters = []
	for l in neighbors:
		l = set(l)
		if len(bin_positions) in l:
			l.remove(len(bin_positions))
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

# function to cluster all atoms over a specified beta phi factor
def cluster_atoms_over(bin_int):

	# create list of positions of all atoms over specified beta phi factor
	bin_ind = np.where(inds > bin_int)[0].tolist()
	if len(bin_ind) == 0:
		return 0
	bin_positions = [positions[i] for i in bin_ind]

	# create KDTree and determine neighbors of each point in the bin
	bin_tree = spatial.cKDTree(bin_positions)
	distances, neighbors = bin_tree.query(bin_positions, k=len(bin_positions), distance_upper_bound=4)

	# create list of sets
	clusters = []
	for l in neighbors:
		l = set(l)
		if len(bin_positions) in l:
			l.remove(len(bin_positions))
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

# function to determine # of atoms that have beta phi value less than given beta phi value, # of clusters, and
# distribution of cluster sizes

def clustering_info_under(bin_int):
	beta_phi_value = phi_star_bins[bin_int]

	# # number of atoms that have beta phi value less than given beta phi value
	# num_atoms = len(u.atoms[(phi_star < beta_phi_value) & (phi_star > 0)])

	# number of clusters that have atoms with beta phi value less than given beta phi value
	# create dictionary with keys that give size of cluster and values that give number of clusters
	num_clusters = 0
	cluster_dict = {}
	max_bin = np.where(phi_star_bins == beta_phi_value)[0][0]
	
	total_clusters = cluster_atoms_under(max_bin)
	if total_clusters == 0:
		return 0, 0, {}, [], []
	num_clusters = len(total_clusters)

	singleton_ind = []
	doubles_ind = []
	triples_ind = []
	bin_positions_ind = np.where(inds < (bin_int))[0]
	bin_positions = [positions[i] for i in bin_positions_ind]
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
				point = [bin_positions[ind][0], bin_positions[ind][1], bin_positions[ind][2]]
				index = positions.index(point)
				singleton_ind.append(index)
		elif len(cluster) == 2:
			for ind in cluster:
				point = [bin_positions[ind][0], bin_positions[ind][1], bin_positions[ind][2]]
				index = positions.index(point)
				doubles_ind.append(index)
		elif len(cluster) == 3:
			for ind in cluster:
				point = [bin_positions[ind][0], bin_positions[ind][1], bin_positions[ind][2]]
				index = positions.index(point)
				triples_ind.append(index)
		else:
			continue

	return num_clusters, cluster_dict, singleton_ind, doubles_ind, triples_ind


def clustering_info_over(bin_int):
	beta_phi_value = phi_star_bins[bin_int]

	# number of atoms that have beta phi value greater than given beta phi value
	num_atoms = len(u.atoms[(phi_star > beta_phi_value)])

	# number of clusters that have atoms with beta phi value greater than given beta phi value
	# create dictionary with keys that give size of cluster and values that give number of clusters
	num_clusters = 0
	cluster_dict = {}
	max_bin = np.where(phi_star_bins == beta_phi_value)[0][0]
	
	total_clusters = cluster_atoms_over(max_bin)
	if total_clusters == 0:
		return 0, 0, {}, [], []
	num_clusters = len(total_clusters)
	singleton_ind = []
	doubles_ind = []
	bin_positions_ind = np.where(inds > (bin_int))[0]
	bin_positions = [positions[i] for i in bin_positions_ind]
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
				point = [bin_positions[ind][0], bin_positions[ind][1], bin_positions[ind][2]]
				index = positions.index(point)
				singleton_ind.append(index)
		elif len(cluster) == 2:
			for ind in cluster:
				point = [bin_positions[ind][0], bin_positions[ind][1], bin_positions[ind][2]]
				index = positions.index(point)
				doubles_ind.append(index)
		else:
			continue

	return num_atoms, num_clusters, cluster_dict, singleton_ind, doubles_ind

# create list of Atom Groups

# ag_singleton_under = []
# ag_doubles_under = []
# ag_singleton_over = []
# ag_doubles_over = []

# for n in range(1, len(hist)):
# 	print(phi_star_bins[n])
# 	num_atoms, num_clusters, cluster_dict, singleton_ind, doubles_ind = clustering_info_under(n + 1)
# 	print("under: ")
# 	print(cluster_dict)

# 	ag_singleton = mda.AtomGroup(singleton_ind, u)
# 	ag_doubles = mda.AtomGroup(doubles_ind, u)
# 	ag_singleton_under.append(ag_singleton)
# 	ag_doubles_under.append(ag_doubles)

# 	num_atoms_over, num_clusters_over, cluster_dict_over, singleton_ind_over, doubles_ind_over = clustering_info_over(n + 1)
# 	print("over: ")
# 	print(cluster_dict_over)

# 	ag_singleton1 = mda.AtomGroup(singleton_ind_over, u)
# 	ag_doubles1 = mda.AtomGroup(doubles_ind_over, u)
# 	ag_singleton_over.append(ag_singleton1)
# 	ag_doubles_over.append(ag_doubles1)

# function to get Atom Groups representing the clusters for certain beta phi value 

atom_groups_clusters_under = []
atom_groups_clusters_over = []
def get_atom_groups(bin_int):
	for cluster in cluster_atoms_under(bin_int):
		atom_group_inds = []
		bin_positions_ind = np.where(inds < bin_int)[0]
		bin_positions = [positions[i] for i in bin_positions_ind]
		for ind in cluster: 
			point = [bin_positions[ind][0], bin_positions[ind][1], bin_positions[ind][2]]
			index = positions.index(point)
			atom_group_inds.append(index)
		atom_group = mda.AtomGroup(atom_group_inds, u)
		atom_groups_clusters_under.append(atom_group)
	for cluster in cluster_atoms_over(bin_int):
		atom_group_inds = []
		bin_positions_ind = np.where(inds > bin_int)[0]
		bin_positions = [positions[i] for i in bin_positions_ind]
		for ind in cluster: 
			point = [bin_positions[ind][0], bin_positions[ind][1], bin_positions[ind][2]]
			index = positions.index(point)
			atom_group_inds.append(index)
		atom_group = mda.AtomGroup(atom_group_inds, u)
		atom_groups_clusters_over.append(atom_group)

# write selections to vmd file

# with mda.selections.vmd.SelectionWriter('selection_culm.vmd', mode='w') as vmd:
# 	cluster_num = 0
# 	for ag in atom_groups_clusters_under:
# 		if len(ag) > 10:
# 			vmd.write(ag, number=None, name='cluster_under.%d' % (cluster_num), frame=None)
# 			cluster_num += 1
# 		else:
# 			continue 

# 	cluster_num = 0
# 	for ag in atom_groups_clusters_over:
# 		if len(ag) > 10:
# 			vmd.write(ag, number=None, name='cluster_over.%d' % (cluster_num), frame=None)
# 			cluster_num += 1
# 		else:
# 			continue 

# 	cluster_num = 0
# 	for ag in ag_singleton_under:
# 		if len(ag) > 0:
# 			vmd.write(ag, number=None, name='singleton_under.%d' % (cluster_num), frame=None)
# 			cluster_num += 1
# 		else:
# 			cluster_num += 1

# 	cluster_num = 0
# 	for ag in ag_singleton_over:
# 		if len(ag) > 0:
# 			vmd.write(ag, number=None, name='singleton_over.%d' % (cluster_num), frame=None)
# 			cluster_num += 1
# 		else:
# 			cluster_num += 1

# 	cluster_num = 0
# 	for ag in ag_doubles_under:
# 		if len(ag) > 0:
# 			vmd.write(ag, number=None, name='doubles_under.%d' % (cluster_num), frame=None)
# 			cluster_num += 1
# 		else:
# 			cluster_num += 1

# 	cluster_num = 0
# 	for ag in ag_doubles_over:
# 		if len(ag) > 0:
# 			vmd.write(ag, number=None, name='doubles_over.%d' % (cluster_num), frame=None)
# 			cluster_num += 1
# 		else:
# 			cluster_num += 1


# visualize distribution of cluster sizes that have atoms with beta phi values less than specified beta phi value

def visualize_dist_under(bin_int):
	beta_phi_value = phi_star_bins[bin_int]
	num_atoms, num_clusters, cluster_dict, singleton_ind, doubles_ind = clustering_info_under(bin_int)
	font = {'size': 10}
	plt.bar(list(cluster_dict.keys()), cluster_dict.values())
	plt.ylim(0, 24)
	plt.xlim(0, 48)
	plt.xlabel("Cluster Sizes")
	plt.ylabel("Number of Clusters")
	plt.xticks(np.arange(0, 48, 2.0))
	plt.yticks(np.arange(0, 24, 2.0))
	plt.title(r'Distribution of Cluster Sizes with $\mathrm{\beta \phi}$ Value less than %f' % (beta_phi_value), fontdict=font)
	plt.show()

def visualize_dist_over(bin_int):
	beta_phi_value = phi_star_bins[bin_int]
	num_atoms, num_clusters, cluster_dict, singleton_ind, doubles_ind = clustering_info_over(bin_int)
	font = {'size': 10}
	plt.bar(list(cluster_dict.keys()), cluster_dict.values())
	plt.ylim(0, 24)
	plt.xlim(0, 48)
	plt.xlabel("Cluster Sizes")
	plt.ylabel("Number of Clusters")
	plt.xticks(np.arange(0, 48, 2.0))
	plt.yticks(np.arange(0, 24, 2.0))
	plt.title(r'Distribution of Cluster Sizes with $\mathrm{\beta \phi}$ Value greater than %f' % (beta_phi_value), fontdict=font)
	plt.show()

