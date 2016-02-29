import numpy as np
import matplotlib.pyplot as plt


data, cleaned = [], []

topic_ids=(20932, 20488, 20910, 20958, 20714, 20636, 20956, 20424, 20916, 20542, 20778, 20690, 20696, 20694, 20832, 20962, 20812, 20814, 20704, 20780, 20766, 20764, 20642, 20686, 20976, 20972, 20584, 20996)

topic_wise_data = {}

with open('accuracy.csv', 'r') as f:
	data = f.readlines()
	cleaned = map(lambda x: eval(x.strip().split()[0]), filter(lambda x: "NA" not in x,data[1:]))

unique_methods = list(set([x[3] for x in cleaned]))


for element in cleaned:
	try:
		# TopicID -> Method, Accuracy, VotesPerDoc
		topic_wise_data[eval(element[4])].append((element[3], element[5], element[6]))
	except KeyError:
		topic_wise_data[eval(element[4])] = [(element[3], element[5], element[6])]


def get_data_for_one_topic(topic, cleaned):
	Y_axis = list(np.arange(0.0,1.2,0.05))

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
			#print items, Y_axis[i], Y_axis[i+1]
			if (items[1] >= Y_axis[i]) and (items[1] < Y_axis[i+1]):
				curr_sum.append(items[0])
				continue
			else:
				mean = np.mean(curr_sum)
				bucketed[method].append(mean)
				curr_sum = [items[0]]
				i += 1
				

	#radius = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	#area = [3.14159, 12.56636, 28.27431, 50.26544, 78.53975, 113.09724]
	#square = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0]

	styles = ['r', 'b', 'g']
	for i, method in enumerate(bucketed):
		Y_axis = Y_axis[0:len(bucketed[method])]
		plt.plot(Y_axis, bucketed[method], color=styles[i], linestyle='--', marker='o', label=method)
	plt.xlabel('VotesPerDoc')
	plt.ylabel('Accuracy')
	plt.title('VotesPerDoc vs Accuracy for topic - %s.' %(topic))
	plt.legend(loc='lower right' )
	plt.show()

	for method in bucketed:
		print method
		for elements in bucketed[method]:
			print elements



# print topic_wise_data
# print unique_methods
for topic in topic_wise_data:
	print topic
	#sorted_cleaned = sorted(topic_wise_data[topic], key = lambda x: x[0])
	get_data_for_one_topic(topic, topic_wise_data[topic])




