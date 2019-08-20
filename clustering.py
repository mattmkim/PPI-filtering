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

# function to cluster atoms in a specified bin
def cluster_atoms(bin_int):
	
	# create list of positions of all atoms in the same bin
	bin_ind = np.where(inds == bin_int)[0].tolist()
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

# cluster atoms in each bin

atom_groups_clusters = []
positions = positions.tolist()
for n in range(len(hist) - 1):
	if hist[n] == 0 or hist[n] == 1:
		n += 1
	else:
		for cluster in cluster_atoms(n + 1):
			atom_group_inds = []
			bin_positions_ind = np.where(inds == (n + 1))[0]
			bin_positions = [positions[i] for i in bin_positions_ind]
			for ind in cluster: 
				point = [bin_positions[ind][0], bin_positions[ind][1], bin_positions[ind][2]]
				index = positions.index(point)
				atom_group_inds.append(index)
			atom_group = mda.AtomGroup(atom_group_inds, u)
			atom_groups_clusters.append(atom_group)

# create atom groups with atoms in same bin 

atom_groups_bins = []
for i in range(1, max(inds)):
	atom_group_inds = [m for m, n in enumerate(inds) if n == i]
	atom_group = mda.AtomGroup(atom_group_inds, u)
	atom_groups_bins.append(atom_group)

# write selections to vmd file

with mda.selections.vmd.SelectionWriter('selection.vmd', mode='w') as vmd:
	cluster_num = 0
	for ag in atom_groups_clusters:
		if len(ag) > 10:
			vmd.write(ag, number=None, name='cluster.%d' % (cluster_num), frame=None)	
			cluster_num += 1
		else: 
		 	continue

	cluster_num = 1
	for ag in atom_groups_bins:
		if len(ag) <= 10:
			cluster_num += 1
		else: 
			vmd.write(ag, number=None, name='bin.%d' % (cluster_num), frame=None)
			cluster_num += 1

# function to determine # of atoms that have beta phi value less than given beta phi value, # of clusters, and
# distribution of cluster sizes

def clustering_info(bin_int):
	beta_phi_value = phi_star_bins[bin_int]
	# number of atoms that have beta phi value less than given beta phi value
	num_atoms = len(u.atoms[(phi_star < beta_phi_value) & (phi_star > 0)])

	# number of clusters that have atoms with beta phi value less than given beta phi value
	# create dictionary with keys that give size of cluster and values that give number of clusters
	num_clusters = 0
	cluster_dict = {}
	max_bin = np.where(phi_star_bins == beta_phi_value)[0][0]
	
	if hist[max_bin - 1] == 0:
		pass
	elif hist[max_bin - 1] == 1:
		num_clusters += 1
	else: 
		num_clusters += len(cluster_atoms(max_bin))

	for cluster in cluster_atoms(max_bin):
		if len(cluster_atoms(max_bin)) == 0:
			continue
		else: 
			if len(cluster) in cluster_dict:
				cluster_dict[len(cluster)] += 1
			else: 
				cluster_dict[len(cluster)] = 1


	return num_atoms, num_clusters, cluster_dict

# visualize distribution of cluster sizes that have atoms with beta phi values less than specified beta phi value

def visualize_dist(bin_int):
	beta_phi_value = phi_star_bins[bin_int]
	num_atoms, num_clusters, cluster_dict = clustering_info(bin_int)
	font = {'size': 10}
	plt.bar(list(cluster_dict.keys()), cluster_dict.values())
	plt.ylim(0, 24)
	plt.xlim(0, 48)
	plt.xlabel("Cluster Sizes")
	plt.ylabel("Number of Clusters")
	plt.xticks(np.arange(0, 48, 2.0))
	plt.yticks(np.arange(0, 24, 2.0))
	plt.title(r'Distribution of Cluster Sizes with $\mathrm{\beta \phi}$ Value between %f and %f' % (phi_star_bins[bin_int - 1], beta_phi_value), fontdict=font)
	plt.show()

