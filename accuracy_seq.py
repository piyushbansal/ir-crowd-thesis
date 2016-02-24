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
   print choices
   total = sum(w for c, w in choices)
   r = random.uniform(0, total)
   upto = 0
   for i, (c, w) in enumerate(choices):
      if upto + w >= r:
         return i
      upto += w
   assert False, "Shouldn't get here"

def get_weighted_sample(elements, probs):
  sum_probs = sum(probs)
  probs  = map(lambda x: x/sum_probs, probs)
  return weighted_choice(zip(elements, probs))

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
      print "Sampling vote number ", index

      # Draw one vote for a random document
      if curr_doc_selected is None:
        updated_doc_idx = random.randrange(len(vote_lists))
        if not unknown_votes[updated_doc_idx]:
          # We ran out of votes for this document, diregard this sequence
          return None
        vote = unknown_votes[updated_doc_idx].pop()
        known_votes[updated_doc_idx].append(vote)
      else:   
        #print "Selected doc number ", curr_doc_selected
        try:    
          vote = unknown_votes[curr_doc_selected].pop()
        except IndexError:
          vote = bool(random.randint(0,1))
        known_votes[curr_doc_selected].append(vote)
        print "Known votes ", known_votes 
      estimates = estimator(texts, known_votes, X, text_similarity, *args)
      #Just need to get the document index, which is element[0] for enumerate(estimates)
      objects = list(enumerate(estimates))
      print "estimates ", objects
      curr_doc_selected = get_weighted_sample(objects,[x[1][1] for x in objects])
      print curr_doc_selected
      # Calculate all the estimates
      try:
        estimates = estimator(texts, known_votes, X, text_similarity, *args)
        labels = [x[0] for x in estimates]
        accuracy_sequences[estimator_name].append(get_accuracy(labels, truths))
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
      sequences = sampler(estimator_dict, sequence_length, texts, vote_lists, truths, X, text_similarity)

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

  N_SEQS_PER_EST = 15

  print_accuracy_sequences_to_stderr({
       'GP' : (est_gp, []),
       'MV' : (est_majority_vote, []),
       #'MEV(1)' : (est_merge_enough_votes, [ 1 ]),
       #'MVNN(0.5)' : (est_majority_vote_with_nn, [ 0.5 ]),
       #'ActiveMergeEnoughVotes(0.2)' : (est_active_merge_enough_votes, [0.2]),
       #'ActiveMergeEnoughVotes(0.1)' : (est_active_merge_enough_votes, [0.1]),
  }, (0.01, 1.05), topic_id, N_SEQS_PER_EST)

  print_accuracy_sequences_to_stderr({
      'ActiveGPVariance' : (est_gp_min_variance, [None]),
    }, (0.01, 1.05), topic_id, N_SEQS_PER_EST, sampler=sample_gp_variance_min_entropy)

