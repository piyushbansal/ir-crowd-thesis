import data
from sklearn.manifold import TSNE
from scipy import sparse
import numpy
import pickle
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


topic_ids=(20932, 20488, 20910, 20958, 20714, 20636, 20956, 20424, 20916, 20542, 20778, 20690, 20696, 20694, 20832, 20962, 20812, 20814, 20704, 20780, 20766, 20764, 20642, 20686, 20976, 20972, 20584, 20996)

for topic_id in topic_ids:
	texts, vote_lists, truths = data.texts_vote_lists_truths_by_topic_id[str(topic_id)]

	pickle_file = open('../data/vectors.pkl', 'rb')
  	vectors = pickle.load(pickle_file)
  	X = numpy.array(vectors[str(topic_id)]).astype(numpy.double)

	model = TSNE(n_components=2, random_state=0)
	reduced = model.fit_transform(X) 

	X_trues = numpy.array([x for i,x in enumerate(X) if truths[i] == True]) 
	X_falses = numpy.array([x for i,x in enumerate(X) if truths[i] == False])
	X_nones = numpy.array([x for i,x in enumerate(X) if truths[i] == None])
	
	plt.scatter(X_trues[:, 0], X_trues[:, 1], c='g')	
	plt.scatter(X_falses[:, 0], X_falses[:, 1], c='r')	
	plt.scatter(X_nones[:, 0], X_nones[:, 1], c='b')	
	
	plt.savefig(str(topic_id))

