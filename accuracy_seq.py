"""

This calculated accuracy sequences and puts writes them to STDERR
separated by tabs

"""
from experiments import get_accuracy, classify_kde_bayes, est_gp, est_gp_min_variance, est_majority_vote_with_nn_more_confidence, est_majority_vote_with_nn_more_confidence_soft_probs, est_gp_more_confidence,  est_minimise_entropy, est_majority_vote, copy_and_shuffle_sublists, est_active_merge_enough_votes, est_merge_enough_votes, est_majority_vote_with_nn
from data import texts_vote_lists_truths_by_topic_id
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys, pickle
import random
import numpy
from scipy import sparse 
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
import copy
import math

get_mean_vote = lambda vote_list: numpy.mean(vote_list) if vote_list else None

def get_indexes_of_smallest_elements(l):
  """
  >>> get_indexes_of_smallest_elements([1,2,3,1,1,1])
  [0, 3, 4, 5]
  >>> get_indexes_of_smallest_elements([0,2,3,-1,-1,100])
  [3, 4]
  >>> get_indexes_of_smallest_elements([0,0,0,0,0,0])
  [0, 1, 2, 3, 4, 5]
  """
  min_element = min(l)
  return [i for i, el in enumerate(l) if el == min_element ]

def weighted_choice(choices):
   #print choices
   total = sum(w for c, w in choices)
   r = random.uniform(0, total)
   upto = 0
   for i, (c, w) in enumerate(choices):
      if upto + w >= r:
         return c
      upto += w
   assert False, "Shouldn't get here"

def get_weighted_sample(elements, probs):
  sum_probs = sum(probs)
  probs  = map(lambda x: x/sum_probs, probs)
  return weighted_choice(zip(elements, probs))

def get_best_sample(elements, probs):
  sorted_possibilities = sorted(elements, key=lambda x: x[1][1], reverse=True)
  print sorted_possibilities
  return sorted_possibilities[0]

def get_covariance_matrix(matrix_a, matrix_b):
  cov_matrix = []
  for i,elements_a in enumerate(matrix_a):
    this_row = []
    for j,elements_b in enumerate(matrix_b):
       this_row.append(numpy.dot(elements_a, elements_b))
    cov_matrix.append(this_row)
  return numpy.matrix(cov_matrix)

def get_mutual_information_based_best_sample(X, known_votes, possibilities):
  X = X.toarray()
  current_vectors, non_current_vectors = [], []
  possibilities = set([x[0] for x in possibilities])

  for i, sample in enumerate(known_votes):
    if i not in possibilities and len(sample) > 0:
      current_vectors.append(X[i])
    else:
      non_current_vectors.append(X[i])
  
  mutual_information = []
  covariance_matrix = get_covariance_matrix(X, X)
 
  for document_idx in possibilities:
    print document_idx, "Seeing if I should sample this document?  "
    sigma_square_y = covariance_matrix.item((document_idx, document_idx))
    sigma_y_a = get_covariance_matrix([X[document_idx]], current_vectors)
    sigma_a_a = get_covariance_matrix(current_vectors, current_vectors)
    inv_sigma_a_a = numpy.matrix(numpy.linalg.inv(sigma_a_a))
    sigma_a_y = get_covariance_matrix(current_vectors, [X[document_idx]])   

    numerator = sigma_square_y - ((sigma_y_a * inv_sigma_a_a) * sigma_a_y) 

    a_bar = numpy.array(filter(lambda x: repr(x)!= repr(X[document_idx]), non_current_vectors))
    sigma_y_a_bar = get_covariance_matrix([X[document_idx]], a_bar)
    sigma_a_bar_a_bar = get_covariance_matrix(a_bar, a_bar)
    inv_sigma_a_bar_a_bar = numpy.linalg.inv(sigma_a_bar_a_bar)
    sigma_a_bar_y = get_covariance_matrix(a_bar, [X[document_idx]])

    denominator = sigma_square_y - ((sigma_y_a_bar * inv_sigma_a_bar_a_bar) * sigma_a_bar_y)
    mutual_information.append((document_idx, numerator.item(0)/denominator.item(0)))
     
  sorted_mutual_information = sorted(mutual_information, key=lambda x: x[1], reverse=True)
  return sorted_mutual_information[0][0]

def get_covariance_based_best_sample(X, known_votes, possibilities):
  X = X.toarray()
  current_vectors = []
  possibilities = set([x[0] for x in possibilities])
  print known_votes, possibilities

  for i, sample in enumerate(known_votes):
    if i not in possibilities and len(sample) > 0:
      current_vectors.append(X[i])
  joint_entropies = []
  #print current_vectors
  #new_array = [tuple(row) for row in current_vectors]
  #current_vectors = numpy.unique(new_array)
  
  for document_idx in possibilities:
    print document_idx
    candidate_vector, covariance_matrix, det_cov_matrix = None, None, None
    candidate_vector = numpy.append(current_vectors, [X[document_idx]], axis=0)
    print candidate_vector
    covariance_matrix = numpy.cov(candidate_vector) 
    print covariance_matrix
    det_cov_matrix = numpy.linalg.det(covariance_matrix)
    #print det_cov_matrix
    joint_entropies.append((document_idx,math.log(det_cov_matrix )))
    print math.log(det_cov_matrix)
 
  sorted_entropies = sorted(joint_entropies, key=lambda x: x[1])
  print sorted_entropies 
  return sorted_entropies[0][0]

def get_density_based_best_sample(X, known_votes, possibilities):
  total_votes = sum(map(lambda x: len(x), known_votes))
  print total_votes
  X = X.toarray()
  current_vectors = numpy.copy(X)
  #print 'X', X
  #print 'known_votes ', known_votes
  original_docs = len(X)
  possibilities = set([x[0] for x in possibilities])
  #print possibilities

  for i, sample in enumerate(known_votes):
    for k in range(len(sample)):
      current_vectors = numpy.append(current_vectors, [X[i]], axis=0)
  #print 'current_vectors ', current_vectors, len(current_vectors)
  #assert current_vectors != X
  model = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(current_vectors)
  scores = model.score_samples(X)
  
  if (total_votes % 3):
    #Explore low density regions
    sorted_scores = sorted(enumerate(scores), key = lambda x: x[1], reverse=True)
  else:
    #Exploit high density regions 1 times out of 3
    sorted_scores = sorted(enumerate(scores), key = lambda x: x[1])
  #print sorted_scores
  for i in range(original_docs):
    if sorted_scores[i][0] in possibilities:
      #print sorted_scores[i][0]
      return sorted_scores[i][0]
  return None

def get_min_entropy_sample(known_votes, possibilities):
  """In this case, we try to compute the document which minimises the entropy by most.
  We sample both postive and genative vote for it, and see if in the average case, it 
  causes the system entropy to be the least.

  possibilities = [(index, label), (index, label), ...]
  known_votes = [[vote_list_for_document_index_i], [vote_list_for_document_index_i+1], ...]
  """
  system_entropy = get_system_entropy(known_votes)
  print "current_system_entropy ", system_entropy 
  results= []
  for doc_index, label in possibilities:
    print doc_index
    #print known_votes[doc_index]
    known_votes[doc_index].append(True)
    #print known_votes[doc_index]
    pos_label_entropy = get_system_entropy(known_votes)

    known_votes[doc_index][-1] = False
    #print known_votes[doc_index]

    neg_label_entropy = get_system_entropy(known_votes)
    known_votes[doc_index].pop(-1)
    #print known_votes[doc_index]

    avg_system_entropy = (pos_label_entropy+neg_label_entropy)/ 2.0
    
    print avg_system_entropy
    curr_entropy_diff = system_entropy - avg_system_entropy
    
    results.append((doc_index, curr_entropy_diff))
  print results
  results = sorted(results, key=lambda x: x[1], reverse=True)
  
  return results[0][0]
  

def get_system_entropy(vote_lists, base=2,method="SUM"):
  """Computes the entropy of the system base "base", using the method 
  specified as a parameter. SUM means the system entropy is sum 
  of entropies of all individual votes_list for docs.

  Note that the entropy of a system with no information is 1.
  """
  system_entropy = 0.0

  if method == "SUM":
    for vote_list in vote_lists:
      p = get_mean_vote(vote_list)
      if p is not None:
        system_entropy += entropy([p, 1-p], base=base)
      else:
        # In this case, there is no data point, hence no information, hence the system gets an additional entropy of 1.
        system_entropy += 1.0
    return system_entropy
  else:
    raise NotImplementedError

def sample_gp_variance_min_entropy(estimator_dict, n_votes_to_sample, texts,
  vote_lists, truths, X, text_similarity, idx=None, return_final=False, *args):
  """ Randomly sample votes and re-calculate estimates.
  """
  random.seed()

  unknown_votes = copy_and_shuffle_sublists(vote_lists)
  
  accuracy_sequences = {}

  for estimator_name, estimator_args in estimator_dict.iteritems():
    estimator, args = estimator_args
    accuracy_sequences[estimator_name] = []

    known_votes = [ [] for _ in unknown_votes ]

    estimates = [None for _ in vote_lists]

    curr_doc_selected = None
    
    # This is a crowdsourcing procedure
    for index in xrange(n_votes_to_sample):
      #print "Sampling vote number ", index

      # Draw one vote for a random document
      if curr_doc_selected is None:
        updated_doc_idx = random.randrange(len(vote_lists))
        if not unknown_votes[updated_doc_idx]:
          # We ran out of votes for this document, diregard this sequence
          return None
        vote = random.choice(unknown_votes[updated_doc_idx])
        known_votes[updated_doc_idx].append(vote)
      else:   
        #print "Selected doc number ", curr_doc_selected
        try:    
          vote = random.choice(unknown_votes[curr_doc_selected])
        except IndexError:
          # We ran out of votes for this document, disregard this sequence
          return None
        known_votes[curr_doc_selected].append(vote)
        #print "Known votes ", known_votes 
      estimates = estimator(texts, known_votes, X, text_similarity, *args)
      #Just need to get the document index, which is element[0] for enumerate(estimates)
      
      estimates = list(estimates)
      #print len(estimates)
      num_votes_step = sum(map(lambda x: len(x), known_votes))/ len(unknown_votes)
      #print num_votes_step
      possibilities = filter(lambda x: len(known_votes[x[0]]) < 1 + num_votes_step ,enumerate(estimates))
      #print possibilities, len(possibilities), list(enumerate(estimates))
      #Just need to get the document index, which is element[0] for enumerate(estimates)
      try:
        curr_doc_selected = get_density_based_best_sample(X, known_votes, possibilities)
        #curr_doc_selected = get_best_sample(possibilities,[x[1][1] for x in possibilities])[0]
        #curr_doc_selected = get_weighted_sample(possibilities,[x[1][1] for x in possibilities])[0]
      except:
        #print "Excepted"
        curr_doc_selected = get_density_based_best_sample(X, known_votes, enumerate(estimates))
        #curr_doc_selected = get_best_sample(enumerate(estimates),[x[1] for x in estimates])[0]
        #curr_doc_selected = get_weighted_sample(enumerate(estimates),[x[1] for x in estimates])[0]
      #print curr_doc_selected

      #objects = list(enumerate(estimates))
      #print "estimates ", objects
      #curr_doc_selected = get_weighted_sample(objects,[x[1][1] for x in objects])
      #print curr_doc_selected
      # Calculate all the estimates
      estimates = estimator(texts, known_votes, X, text_similarity, *args)
      labels = [x[0] for x in estimates]
      try:
        accuracy = get_accuracy(labels, truths)
        accuracy_sequences[estimator_name].append(accuracy)
      except Exception, e:
        accuracy_sequences[estimator_name].append(None)
  
  return accuracy_sequences

def sample_min_entropy(estimator_dict, n_votes_to_sample, texts,
  vote_lists, truths, X, text_similarity, idx=None, return_final=False, *args):
  """ Randomly sample votes and re-calculate estimates.
  """
  random.seed()

  unknown_votes = copy_and_shuffle_sublists(vote_lists)
  
  accuracy_sequences = {}

  for estimator_name, estimator_args in estimator_dict.iteritems():
    estimator, args = estimator_args
    accuracy_sequences[estimator_name] = []

    known_votes = [ [] for _ in unknown_votes ]

    estimates = [None for _ in vote_lists]

    curr_doc_selected = None

    document_idx_vote_seq = []

    document_vote_counts = [ 0 for _ in vote_lists ]

    #Randomly sampling 30 votes first for avoiding bias etc.
    """for votes_required in range(0):
      min_vote_doc_idxs = get_indexes_of_smallest_elements(document_vote_counts)
      updated_doc_idx = random.choice(min_vote_doc_idxs)
      document_vote_counts[updated_doc_idx] += 1

      # Randomly pick a vote for this document
      vote_idx = random.randrange(len(vote_lists[updated_doc_idx]))

      vote = vote_lists[updated_doc_idx][vote_idx]
      document_idx_vote_seq.append( (updated_doc_idx, vote ) )

    for document_idx, vote in document_idx_vote_seq:
      known_votes[document_idx].append(vote)
      estimates = estimator(texts, known_votes, X, text_similarity, *args)
      try:
        accuracy = get_accuracy(estimates, truths)
        accuracy_sequences[estimator_name].append(accuracy)
      except Exception, e:
        print "Pooped"
        return None
    print "known_votes ", known_votes"""
    # This is a crowdsourcing procedure, random sampling end
    for index in xrange(n_votes_to_sample):
      print "Sampling vote number ", index

      # Draw one vote for a random document
      if curr_doc_selected is None:
        print "Sampling random vote yall"
        updated_doc_idx = random.randrange(len(vote_lists))
        if not unknown_votes[updated_doc_idx]:
          # We ran out of votes for this document, diregard this sequence
          return None
        vote = random.choice(unknown_votes[updated_doc_idx])
        known_votes[updated_doc_idx].append(vote)
      else:   
        print "Selected doc number ", curr_doc_selected
        try:    
          vote = random.choice(unknown_votes[curr_doc_selected])
        except IndexError:
          # We ran out of votes for this document, disregard this sequence
          return None
        known_votes[curr_doc_selected].append(vote)
        #print "Known votes ", known_votes

      if not index % 50:
        # While doing density based sampling, we don't really need to do label aggregation at each point.
        # Still doing it at every 50th vote, just to keep this code around for other 
        # sampling methods like entropy based.
        estimates = estimator(texts, known_votes, X, text_similarity, *args)
      
      estimates = list(estimates)
      print estimates, len(estimates)
      num_votes_step = sum(map(lambda x: len(x), known_votes))/ len(unknown_votes)
      print 'num_vote_step ', num_votes_step
      possibilities = filter(lambda x: len(known_votes[x[0]]) < 1 + num_votes_step ,enumerate(known_votes))
      #print possibilities, len(possibilities), list(enumerate(estimates))
      #Just need to get the document index, which is element[0] for enumerate(estimates)
      try:
        #curr_doc_selected = get_best_sample(possibilities,[x[1][1] for x in possibilities])[0]
        #curr_doc_selected = get_weighted_sample(possibilities,[x[1][1] for x in possibilities])[0]
        #curr_doc_selected = get_min_entropy_sample(known_votes, possibilities)
        #curr_doc_selected = get_density_based_best_sample(X, known_votes, possibilities)
        #curr_doc_selected = get_covariance_based_best_sample(X, known_votes, possibilities) 
        curr_doc_selected = get_mutual_information_based_best_sample(X, known_votes, possibilities) 
      except Exception as e:
        print "Excepted", e
        #curr_doc_selected = get_best_sample(enumerate(estimates),[x[1] for x in estimates])[0]
        #curr_doc_selected = get_weighted_sample(enumerate(estimates),[x[1] for x in estimates])[0]
        #curr_doc_selected = get_min_entropy_sample(known_votes, enumerate(estimates))
        curr_doc_selected = get_density_based_best_sample(X, known_votes, enumerate(estimates))
        
      #print "Curr_doc_selected ", curr_doc_selected

      #objects = list(enumerate(estimates))
      #print "estimates ", objects
      #curr_doc_selected = get_weighted_sample(objects,[x[1][1] for x in objects])
      #print curr_doc_selected
      # Calculate all the estimates
      estimates = estimator(texts, known_votes, X, text_similarity, *args)
      #labels = [x[0] for x in estimates]
      try:
        accuracy = get_accuracy(estimates, truths)
        accuracy_sequences[estimator_name].append(accuracy)
      except Exception, e:
        return None
        #accuracy_sequences[estimator_name].append(None)
  
  return accuracy_sequences

def sample_min_entropy_kde(estimator_dict, start_idx, n_votes_to_sample, texts,
  vote_lists, truths, X, text_similarity, idx=None, return_final=False, *args):
  """ Randomly sample votes and re-calculate estimates.
  """
  random.seed()

  unknown_votes = copy_and_shuffle_sublists(vote_lists)
  
  accuracy_sequences = {}

  for estimator_name, estimator_args in estimator_dict.iteritems():
    estimator, args = estimator_args
    accuracy_sequences[estimator_name] = []

    known_votes = [ [] for _ in unknown_votes ]

    estimates = [None for _ in vote_lists]

    curr_doc_selected = None

    document_idx_vote_seq = []

    document_vote_counts = [ 0 for _ in vote_lists ]

    for votes_required in range(start_idx):
      min_vote_doc_idxs = get_indexes_of_smallest_elements(document_vote_counts)
      updated_doc_idx = random.choice(min_vote_doc_idxs)
      document_vote_counts[updated_doc_idx] += 1

      # Randomly pick a vote for this document
      vote_idx = random.randrange(len(vote_lists[updated_doc_idx]))

      vote = vote_lists[updated_doc_idx][vote_idx]
      document_idx_vote_seq.append( (updated_doc_idx, vote ) )

    for document_idx, vote in document_idx_vote_seq:
      known_votes[document_idx].append(vote)
    print n_votes_to_sample
 
    # This is a crowdsourcing procedure
    for index in xrange(n_votes_to_sample):
      print "Sampling vote number ", index

      # Draw one vote for a random document
      if curr_doc_selected is None:
        updated_doc_idx = random.randrange(len(vote_lists))
        if not unknown_votes[updated_doc_idx]:
          # We ran out of votes for this document, diregard this sequence
          return None
        vote = random.choice(unknown_votes[updated_doc_idx])
        known_votes[updated_doc_idx].append(vote)
      else:   
        #print "Selected doc number ", curr_doc_selected
        try:    
          vote = random.choice(unknown_votes[curr_doc_selected])
        except IndexError:
          # We ran out of votes for this document, disregard this sequence
          return None
        known_votes[curr_doc_selected].append(vote)
        print "Known votes ", known_votes 
      estimates = estimator(texts, known_votes, X, text_similarity, *args)
      #Just need to get the document index, which is element[0] for enumerate(estimates)
      
      estimates = list(estimates)
      #print len(estimates)
      num_votes_step = sum(map(lambda x: len(x), known_votes))/ len(unknown_votes)
      #print num_votes_step
      possibilities = filter(lambda x: len(known_votes[x[0]]) < 1 + num_votes_step ,enumerate(estimates))
      #print possibilities, len(possibilities), list(enumerate(estimates))
      #Just need to get the document index, which is element[0] for enumerate(estimates)
      try:
        #curr_doc_selected = get_best_sample(possibilities,[x[1][1] for x in possibilities])[0]
        #curr_doc_selected = get_weighted_sample(possibilities,[x[1][1] for x in possibilities])[0]
        curr_doc_selected = get_min_entropy_sample(known_votes, possibilities)
      except:
        #print "Excepted"
        #curr_doc_selected = get_best_sample(enumerate(estimates),[x[1] for x in estimates])[0]
        #curr_doc_selected = get_weighted_sample(enumerate(estimates),[x[1] for x in estimates])[0]
        curr_doc_selected = get_min_entropy_sample(known_votes, enumerate(estimates))
      #print "Curr_doc_selected ", curr_doc_selected

      #objects = list(enumerate(estimates))
      #print "estimates ", objects
      #curr_doc_selected = get_weighted_sample(objects,[x[1][1] for x in objects])
      #print curr_doc_selected
      # Calculate all the estimates
      estimates = estimator(texts, known_votes, X, text_similarity, *args)
      #labels = [x[0] for x in estimates]
      try:
        accuracy = get_accuracy(estimates, truths)
        accuracy_sequences[estimator_name].append(accuracy)
      except Exception, e:
        accuracy_sequences[estimator_name].append(None)
  
  return accuracy_sequences  


def get_accuracy_sequences(estimator_dict, sequence_length, texts, vote_lists, truths, X, text_similarity):

  random.seed() # This is using system time

  document_idx_vote_seq = []

  document_vote_counts = [ 0 for _ in vote_lists ]

  # Conduct an experiment where you randomly sample votes for documents
  for _ in xrange(sequence_length):
    # Pick a document randomly from the ones that has fewer votes
    min_vote_doc_idxs = get_indexes_of_smallest_elements(document_vote_counts)
    updated_doc_idx = random.choice(min_vote_doc_idxs)
    document_vote_counts[updated_doc_idx] += 1

    # Randomly pick a vote for this document
    vote_idx = random.randrange(len(vote_lists[updated_doc_idx]))

    vote = vote_lists[updated_doc_idx][vote_idx]
    document_idx_vote_seq.append( (updated_doc_idx, vote ) )

  # Here we know the sequence of draws was successful
  # Let us measure estimator accuracies now
  accuracy_sequences = {}

  for estimator_name, estimator_args in estimator_dict.iteritems():
    estimator, args = estimator_args
    accuracy_sequences[estimator_name] = []

    # Go through the generated sequence of draws and measure accuracy
    known_votes = [ [] for _ in vote_lists ]
    #i = 0
    for document_idx, vote in document_idx_vote_seq:
      known_votes[document_idx].append(vote)
      #i += 1
      #if i < 30:
      #  continue
      
      # Recalculate all the estimates for the sake of consistency
      estimates = estimator(texts, known_votes, X, text_similarity, *args)

      # Calucate the accuracy_sequence
      try:
        accuracy =  get_accuracy(estimates, truths)
      except OSError:
        print '#OS ERROR'
        # Leave the function
        return None

      accuracy_sequences[estimator_name].append(accuracy)
  
  return accuracy_sequences

def print_accuracy_sequences_to_stderr(estimator_dict, votes_per_doc, topic_id, n_sequesnces_per_estimator, sampler=get_accuracy_sequences):
  texts, vote_lists, truths = texts_vote_lists_truths_by_topic_id[topic_id]
  n_documents = len(texts)

  pickle_file = open('../data/vectors.pkl', 'rb')
  vectors = pickle.load(pickle_file)
  X = sparse.csr_matrix(numpy.array(vectors[topic_id]).astype(numpy.double))

  text_similarity = cosine_similarity(X)

  min_votes_per_doc, max_votes_per_doc = votes_per_doc

  start_vote_count = int(min_votes_per_doc * n_documents)
  # In an accuracy sequence, element 0 corresponds to the vote count of 1.
  start_idx = start_vote_count - 1

  sequence_length = int(max_votes_per_doc * n_documents)

  for _ in xrange(n_sequesnces_per_estimator):
    # Getting accuracy for all esimators
    # If failed, attempt at getting a sequence until it's not None
    sequences = None
    counter = 0
    while sequences is None:
      counter += 1
      print '#ATTEMPT\t%s' % counter
      if sampler != sample_min_entropy_kde:
        sequences = sampler(estimator_dict, sequence_length, texts, vote_lists, truths, X, text_similarity)
      elif sampler == sample_min_entropy_kde:
        sequence_length = sequence_length - start_idx
        sequences = sampler(estimator_dict, start_idx, sequence_length, texts, vote_lists, truths, X, text_similarity)

    # Got a sequence
    # Write all sequences from this dict to stderr
    run_id = random.randint(0, sys.maxint)

    for estimator_name, accuracy_sequence in sequences.iteritems():
      accuracy_sequence_trimmed = accuracy_sequence[start_idx: ]
      
      for index, accuracy in enumerate(accuracy_sequence_trimmed):
        sys.stderr.write("AC\t%s\t%s\t%s\t%s\t%s\n" % (start_vote_count + index, run_id, estimator_name, topic_id, "%.4f" % accuracy) )


if __name__ == "__main__":
  try:
    topic_id = sys.argv[1]
  except IndexError:
    raise Exception("Please supply the topic id")

  N_SEQS_PER_EST = 3

  print_accuracy_sequences_to_stderr({
       #'GP' : (est_gp, []),
       #'MV' : (est_majority_vote, []),
       #'KDE': (classify_kde_bayes, [None]),
       #'MEV(3)' : (est_merge_enough_votes, [ 3 ]),
       #'MVNN(0.5)' : (est_majority_vote_with_nn, [ 0.5 ]),
       #'ActiveMergeEnoughVotes(0.2)' : (est_active_merge_enough_votes, [0.2]),
       #'ActiveMergeEnoughVotes(0.1)' : (est_active_merge_enough_votes, [0.1]),
  }, (0.01, 1.05), topic_id, N_SEQS_PER_EST)

  print_accuracy_sequences_to_stderr({
      #'ActiveGPVariance' : (est_gp_min_variance, [None]),
    }, (0.01, 3.05), topic_id, N_SEQS_PER_EST, sampler=sample_gp_variance_min_entropy)

  print_accuracy_sequences_to_stderr({
      'ActiveGPMutualInformation' : (est_gp, []),
      #'MV' : (est_majority_vote, []),
      #'ActiveMEV(3)' : (est_merge_enough_votes, [ 3 ]),
      #'ActiveMVNN(0.5)' : (est_majority_vote_with_nn, [ 0.5 ]),
      #'ActiveKDE': (classify_kde_bayes, [None]),
    }, (0.01, 1.05), topic_id, N_SEQS_PER_EST, sampler=sample_min_entropy)

  print_accuracy_sequences_to_stderr({
      #'MV' : (est_majority_vote, []),
      #'ActiveMEV(3)' : (est_merge_enough_votes, [ 3 ]),
      #'ActiveMVNN(0.3)' : (est_majority_vote_with_nn, [ 0.3 ]),
      #'KDE': (classify_kde_bayes, [None]),
    }, (0.2, 3.05), topic_id, N_SEQS_PER_EST, sampler=sample_min_entropy_kde)
