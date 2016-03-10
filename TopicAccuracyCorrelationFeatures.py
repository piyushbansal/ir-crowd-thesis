import data
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


data_rows, cleaned = [], []

topic_ids=(20932, 20488, 20910, 20958, 20714, 20636, 20956, 20424, 20916, 20542, 20778, 20690, 20696, 20694, 20832, 20962, 20812, 20814, 20704, 20780, 20766, 20764, 20642, 20686, 20976, 20972, 20584, 20996)

topic_wise_data = {}

with open('/Users/piyushbansal/accuracy.csv', 'r') as f:
	data_rows = f.readlines()
	cleaned = map(lambda x: eval(x.strip().split()[0]), filter(lambda x: "NA" not in x,data_rows[1:]))

unique_methods = list(set([x[3] for x in cleaned]))


for element in cleaned:
	try:
		# TopicID -> Method, Accuracy, VotesPerDoc
		topic_wise_data[eval(element[4])].append((element[3], element[5], element[6]))
	except KeyError:
		topic_wise_data[eval(element[4])] = [(element[3], element[5], element[6])]


def get_data_for_one_topic(topic, cleaned):
	Y_axis = np.arange(0.0,3.1,0.05)

	all_rows = {}

	for method in unique_methods:
		all_rows[method] = map(lambda x: (x[1],x[2]), filter(lambda x: method == x[0], cleaned))
		all_rows[method] = sorted(all_rows[method], key=lambda x: x[1])
    
	bucketed = {}
	for method in unique_methods:
		bucketed[method] = []

	for method in unique_methods:
		i = 0
		curr_sum = []
		for items in all_rows[method]:
			if (items[1] >= Y_axis[i]) and (items[1] < Y_axis[i+1]):
				curr_sum.append(items[0])
				continue
			else:
				mean = np.mean(curr_sum)
				bucketed[method].append(mean)
				curr_sum = [items[0]]
				i += 1
				
	return bucketed

	
topic_wise_features_data = {}

features = ['rel_irrel_ratio', 'avg_inner_similarity', 'avg_outer_similarity', 'inner_outer_similarity_ratio', 'num_docs', 'avg_votes', 'ground_truth_relevance', 'avg_known_label_docs']

for topic_id in topic_ids:
	topic_wise_features_data[topic_id] = {}

	texts, vote_lists, truths = data.texts_vote_lists_truths_by_topic_id[str(topic_id)]

	pickle_file = open('/Users/piyushbansal/vectors.pkl', 'rb')
  	vectors = pickle.load(pickle_file)
  	X = np.array(vectors[str(topic_id)]).astype(np.double)

  	#Compute accuracies
  	accuracies = get_data_for_one_topic(topic_id, topic_wise_data[topic_id])
  	topic_wise_features_data[topic_id]['accuracies'] = accuracies

  	#Compute rel_irrel_ratio
  	rel_labels = sum([1 for i,x in enumerate(X) if truths[i] == True]) * 1.0
  	irrel_labels = sum([1 for i,x in enumerate(X) if truths[i] == False]) *1.0
  	topic_wise_features_data[topic_id]['rel_irrel_ratio'] = rel_labels/irrel_labels

  	#Compute avg_inner_similarity, avg_outer_similarity, inner_outer_similarity_ratio
  	rel_docs = np.array([x for i,x in enumerate(X) if truths[i] == True])
	irrel_docs = np.array([x for i,x in enumerate(X) if truths[i] == False])

	avg_inner_similarities = np.mean((filter(lambda x: x<= 0.99 and x>= 0.0001, cosine_similarity(rel_docs).flatten())))
	topic_wise_features_data[topic_id]['avg_inner_similarity'] = avg_inner_similarities

	avg_outer_similarities = np.mean((filter(lambda x: x<= 0.99 and x>= 0.0001, cosine_similarity(rel_docs, irrel_docs).flatten())))
	topic_wise_features_data[topic_id]['avg_outer_similarity'] = avg_outer_similarities

	inner_outer_similarity_ratio = avg_inner_similarities/ avg_outer_similarities
	topic_wise_features_data[topic_id]['inner_outer_similarity_ratio'] = inner_outer_similarity_ratio

	#Compute num_docs
	topic_wise_features_data[topic_id]['num_docs'] = len(X)

	#Compute avg_votes
	#Number of votes per document on an average.
	topic_wise_features_data[topic_id]['avg_votes'] = np.mean([len(x) for x in vote_lists])

	#Compute ground_truth_relevance
	#Proportion of votes that match with ground truth
	filtered_vote_list_truths = [(x,truths[i]) for i,x in enumerate(vote_lists) if truths[i] != None]
	topic_wise_features_data[topic_id]['ground_truth_relevance'] = np.mean([np.mean([1 if x == label else 0 for x in list]) for (list,label) in filtered_vote_list_truths])

	#Compute known_label_docs
	topic_wise_features_data[topic_id]['avg_known_label_docs'] = np.mean([1 if (truths[i] == True or truths[i] == False) else 0 for i,x in enumerate(X)])

	
selected_topic_ids = [topic_id for topic_id in topic_ids if len(topic_wise_features_data[topic_id]['accuracies']['MV']) > 52]

for method in unique_methods:

	method_accuracy_at_point_5 = [topic_wise_features_data[topic_id]['accuracies'][method][11] for topic_id in selected_topic_ids]
	method_accuracy_at_1 = [topic_wise_features_data[topic_id]['accuracies'][method][21] for topic_id in selected_topic_ids]
	method_accuracy_at_1_point_5 = [topic_wise_features_data[topic_id]['accuracies'][method][31] for topic_id in selected_topic_ids]
	method_accuracy_at_2 = [topic_wise_features_data[topic_id]['accuracies'][method][41] for topic_id in selected_topic_ids]
	method_accuracy_at_2_point_5 = [topic_wise_features_data[topic_id]['accuracies'][method][51] for topic_id in selected_topic_ids]

	print method
	for feature in features:
		print feature, pearsonr([topic_wise_features_data[topic_id][feature] for topic_id in selected_topic_ids], method_accuracy_at_point_5)
		print feature, pearsonr([topic_wise_features_data[topic_id][feature] for topic_id in selected_topic_ids], method_accuracy_at_1)
		print feature, pearsonr([topic_wise_features_data[topic_id][feature] for topic_id in selected_topic_ids], method_accuracy_at_1_point_5)
		print feature, pearsonr([topic_wise_features_data[topic_id][feature] for topic_id in selected_topic_ids], method_accuracy_at_2)
		print feature, pearsonr([topic_wise_features_data[topic_id][feature] for topic_id in selected_topic_ids], method_accuracy_at_2_point_5)
	print 



