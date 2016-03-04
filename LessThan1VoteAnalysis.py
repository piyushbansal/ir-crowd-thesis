import numpy as np
import matplotlib.pyplot as plt

Y_axis = list(np.arange(0,1.1,0.05))

data = open('means_old.csv').readlines()
cleaned = map(lambda x: eval(x.strip().split()[0]), data[1:])

unique_methods = list(set([x[1] for x in cleaned]))
#print unique_methods

all_rows = {}

for method in unique_methods:
	all_rows[method] = map(lambda x: (x[2],x[3]), filter(lambda x: method == x[1], cleaned))
	all_rows[method] = sorted(all_rows[method], key=lambda x: x[0])

bucketed = {}
for method in unique_methods:
	bucketed[method] = []

for method in unique_methods:
	i = 0
	curr_sum = []
	for items in all_rows[method]:
		print items, Y_axis[i], Y_axis[i+1]
		if (items[0] >= Y_axis[i]) and (items[0] < Y_axis[i+1]):
			curr_sum.append(items[1])
			continue
		else:
			mean = np.mean(curr_sum)
			bucketed[method].append(mean)
			curr_sum = [items[1]]
			i += 1
			
				


for i in range(21):
	print Y_axis[i]

styles = ['r', 'b', 'g']
for i, method in enumerate(bucketed):
	Y_axis = Y_axis[0:len(bucketed[method])]
	plt.plot(Y_axis, bucketed[method], color=styles[i], linestyle='--', marker='o', label=method)
plt.xlabel('VotesPerDoc')
plt.ylabel('Mean Accuracy')
plt.title('VotesPerDoc vs Mean Accuracy')
plt.legend(loc='lower right' )
plt.show()

for method in bucketed:
	print method
	for elements in bucketed[method]:
		print elements
