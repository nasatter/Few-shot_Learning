import torch
import copy
import torch.nn.functional as F

# This class takes the intial shot guess and the training set
# to optimize the few-shot data set
# Theory: Outliers and miss-classifications included in the few-shot list can 
# 		  dranstically reduce the classifier performance especially when 
#		  few (~1-10) shot is employed. It is expected that by selecting
#		  the few-shot comparison to be within a range of the mean cluster
#		  we can eliminate the outlier inclusion.
# Inputs:
#		data_dict:		dictionary containing the sets
# 
# Outputs: dictionary containing updated sets
#
# Fuctions: 
#		load_batch:    loads batch data and cats to tensor for each class
#		load_shot:     loads shot data and cats to tensor for each class
#		clear_batch:   clears all data
#		optimize_shot: runs clustering and randomly selects candidates
#					   with a specified range	
#		upate_dict: updates the dictionary with new shot guess
#
class optimize_shot():
	def __init__(self, data_dict,num_shot,skip_iter=0,iterations=150, greedy = True):
		self.data_dict = data_dict
		self.num_classes = len(data_dict['train'].keys())
		self.train_clusters = [[]]*self.num_classes
		self.img_names = [[]]*self.num_classes
		self.num_shot = num_shot
		self.iterations = iterations+skip_iter
		self.skip_iter = skip_iter
		self.num_iters = 0
		self.greedy = greedy

	def active(self):
		return ((self.num_iters < self.iterations) and (self.skip_iter<=self.num_iters))


	def load_batch(self,batch_feature,batch_class,img_name):
		if self.active():
			batch_feature = batch_feature.clone().detach().cpu()
			batch_class = batch_class.clone().detach().cpu()
			for i in range(self.num_classes):
				index = (batch_class==i)
				index = index.nonzero()
				index = index[:,0]
				if len(self.train_clusters[i])==0:
					self.train_clusters[i] = batch_feature[index,:]
					#print(img_name,index)
					self.img_names[i] = [img_name[j] for j in index]
				else:
					#print(self.img_names[i],img_name)
					self.train_clusters[i] = torch.cat((self.train_clusters[i],batch_feature[index,:]),dim=0)
					self.img_names[i] += [img_name[j] for j in index]



	def optimize_shot(self):
		if self.active():
			for i in range(self.num_classes):
				all_training=self.train_clusters[i]
				cluster_center = torch.mean(all_training,dim=0)
				distance = F.pairwise_distance(cluster_center,all_training)
				std_dev = torch.std(distance)
				average = torch.mean(distance)
				mx = torch.max(distance)
				mn = torch.min(distance)
				index_class = self.data_dict['index_class'][i]
				keep = self.check_shot(self.img_names[i],self.data_dict['shot'][index_class],distance,std_dev,average)
				if (self.num_shot-len(keep))>0:
					if self.greedy:
						_,candidates = distance.sort()
						perm = candidates[0:(self.num_shot-len(keep))]
					else:
						candidates = torch.lt(distance,average-std_dev/2).nonzero()
						bias = std_dev/2
						while candidates.shape[0] < self.num_shot:
							bias = bias/2
							candidates = torch.lt(distance,average-bias).nonzero()
							#print(candidates.shape[0],bias,average)
							
						perm = torch.randperm(candidates.shape[0])[0:self.num_shot-len(keep)]
					
					selected = [self.img_names[i][j] for j in perm]+keep
					print(mx,mn,std_dev,average,candidates.shape,len(keep),len(selected),len(self.data_dict['shot'][index_class]))
					
					self.data_dict['shot'][index_class]=selected
			self.clear_batch()
		self.num_iters += 1
		return self.data_dict

	def check_shot(self,shot_list,img_list,distance,std_dev,average):
		shot_index = torch.tensor([(i in shot_list)*ind for ind,i in enumerate(img_list)]).nonzero().squeeze()
		check = torch.lt(distance[shot_index],average).nonzero()
		keep = [shot_list[i] for i in check]
		#print(keep)
		return keep

	def clear_batch(self):
		#print(self.clusters)
		self.train_clusters = [[]]*self.num_classes
		self.img_names = [[]]*self.num_classes

