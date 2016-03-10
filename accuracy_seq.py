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
        print "Known votes ", known_votes 
      estimates = estimator(texts, known_votes, X, text_similarity, *args)
      #Just need to get the document index, which is element[0] for enumerate(estimates)
      
      estimates = list(estimates)
      #print len(estimates)
      num_votes_step = sum(map(lambda x: bool(x), known_votes))/ len(unknown_votes)
      print num_votes_step
      possibilities = filter(lambda x: len(known_votes[x[0]]) < 1 + num_votes_step ,enumerate(estimates))
      print possibilities, len(possibilities), list(enumerate(estimates))
      #Just need to get the document index, which is element[0] for enumerate(estimates)
      try:
        curr_doc_selected = get_best_sample(possibilities,[x[1][1] for x in possibilities])[0]
        #curr_doc_selected = get_weighted_sample(possibilities,[x[1][1] for x in possibilities])[0]
      except:
        print "Excepted"
        curr_doc_selected = get_best_sample(enumerate(estimates),[x[1] for x in estimates])[0]
        #curr_doc_selected = get_weighted_sample(enumerate(estimates),[x[1] for x in estimates])[0]
      print curr_doc_selected

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
      num_votes_step = sum(map(lambda x: bool(x), known_votes))/ len(unknown_votes)
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
      num_votes_step = sum(map(lambda x: bool(x), known_votes))/ len(unknown_votes)
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

    for document_idx, vote in document_idx_vote_seq:
      known_votes[document_idx].append(vote)
      
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

  N_SEQS_PER_EST = 25

  print_accuracy_sequences_to_stderr({
       #'GP' : (est_gp, []),
       #'MV' : (est_majority_vote, []),
       #'MEV(3)' : (est_merge_enough_votes, [ 3 ]),
       #'MVNN(0.3)' : (est_majority_vote_with_nn, [ 0.3 ]),
       #'ActiveMergeEnoughVotes(0.2)' : (est_active_merge_enough_votes, [0.2]),
       #'ActiveMergeEnoughVotes(0.1)' : (est_active_merge_enough_votes, [0.1]),
  }, (0.01, 3.05), topic_id, N_SEQS_PER_EST)

  print_accuracy_sequences_to_stderr({
      #'ActiveGPVariance' : (est_gp_min_variance, [None]),
    }, (0.01, 2.05), topic_id, N_SEQS_PER_EST, sampler=sample_gp_variance_min_entropy)

  print_accuracy_sequences_to_stderr({
      #'MV' : (est_majority_vote, []),
      #'ActiveMEV(3)' : (est_merge_enough_votes, [ 3 ]),
      #'ActiveMVNN(0.3)' : (est_majority_vote_with_nn, [ 0.3 ]),
      #'KDE': (classify_kde_bayes, [None]),
    }, (0.01, 3.05), topic_id, N_SEQS_PER_EST, sampler=sample_min_entropy)

  print_accuracy_sequences_to_stderr({
      #'MV' : (est_majority_vote, []),
      #'ActiveMEV(3)' : (est_merge_enough_votes, [ 3 ]),
      #'ActiveMVNN(0.3)' : (est_majority_vote_with_nn, [ 0.3 ]),
      'KDE': (classify_kde_bayes, [None]),
    }, (0.2, 3.05), topic_id, N_SEQS_PER_EST, sampler=sample_min_entropy_kde)



