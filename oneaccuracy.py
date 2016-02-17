from experiments import get_accuracy, est_gp, est_gp_more_confidence,  est_minimise_entropy, est_majority_vote, copy_and_shuffle_sublists, est_active_merge_enough_votes, est_merge_enough_votes, est_majority_vote_with_nn
from data import texts_vote_lists_truths_by_topic_id
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import codecs
import sys
import random
import traceback
import pickle
from scipy import sparse, io
import numpy
from scipy.stats import entropy

get_mean_vote = lambda vote_list: numpy.mean(vote_list) if vote_list else None

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

def add_lambda_votes_to_vote_lists(known_votes, lambda_votes):
  lambda_votes = list(lambda_votes)

  for doc_index, doc_vote_list in enumerate(known_votes):
    vote = lambda_votes[doc_index]
    if vote is not None:
      doc_vote_list.append(lambda_votes[doc_index])
  
  return known_votes


def sample_and_minimise_entropy(final_estimator, estimator, n_votes_to_sample, texts, 
  vote_lists, truths, X, text_similarity, idx=None, return_final=False, *args):
  """ Active learning scenario, where we decide at each step what document to pick.
  Then we ask a worker to label it for us. Our hypothesis is that it should be better
  than randomly asking for labels of some documents. We're only picking docs
  that minimise the uncertainity in system.
  """

  random.seed()

  unknown_votes = copy_and_shuffle_sublists(vote_lists)
  known_votes = [ [] for _ in unknown_votes ]

  estimates = [None for _ in vote_lists]

  random_vote_taken = False
  doc_to_be_sampled = None
  
  #Continue until you can get one vote for some document.
  #This is just some way of initialisation.
  while not random_vote_taken:
    print "Trying to take the first vote"
    updated_doc_idx = random.randrange(len(vote_lists))
    if not unknown_votes[updated_doc_idx]:
      continue
    else:
      #Note that the vote is not replaced here for now. 
      #Since we're dealing with low budget scenario, this should be fine.
      vote = unknown_votes[updated_doc_idx].pop()
      known_votes[updated_doc_idx].append(vote)
      random_vote_taken = True

  print "First vote taken ", updated_doc_idx, vote 

  for index in xrange(n_votes_to_sample):
    # This is how the system entropy looks like, right now.
    # We have just sampled one vote about a random document.
    print "Sampling vote number ", index

    system_entropy = get_system_entropy(known_votes)
    last_iter_entropy = system_entropy
    last_iter_labels = known_votes

    if doc_to_be_sampled is not None:
      print "Document Sampled ", doc_to_be_sampled
      try:
        vote = unknown_votes[doc_to_be_sampled].pop()
        known_votes[doc_to_be_sampled].append(vote)
      except IndexError:
        vote = None
        #TODO(What better can be done? I'm taking a random vote, if no vote is available)
        known_votes[doc_to_be_sampled].append(bool(random.randint(0,1)))

      labels = estimator(texts, known_votes, X, text_similarity, *args)  
      known_votes = add_lambda_votes_to_vote_lists(known_votes, labels)

    # Pick a document that will minimise the system entropy the most
    for doc_index, doc_vote_list in enumerate(known_votes):

      #Adding a positive vote to this document.
      doc_vote_list.append(True)
      
      # At this point, we can either add full votes, or a portion of those.
      # Consider GP, we can add probability which is outputted at the end.
      labels = estimator(texts, known_votes, X, text_similarity, *args)

      known_votes_plus_labels = add_lambda_votes_to_vote_lists(known_votes, labels)

      relevance_label_added_system_entropy = get_system_entropy(known_votes_plus_labels)

      #Adding a negative vote to this document.
      doc_vote_list[-1] = False
      
      # At this point, we can either add full votes, or a portion of those.
      # Consider GP, we can add probability which is outputted at the end.
      labels = estimator(texts, known_votes, X, text_similarity, *args)

      known_votes_plus_labels = add_lambda_votes_to_vote_lists(known_votes, labels)

      non_relevance_label_added_system_entropy = get_system_entropy(known_votes_plus_labels)

      #Calculating the average entropy of the system in both cases.
      doc_avg_system_entropy = (relevance_label_added_system_entropy + non_relevance_label_added_system_entropy) / 2

      #Restore the state of the doc_vote_list.
      doc_vote_list.pop()

      if doc_avg_system_entropy < last_iter_entropy:
        doc_to_be_sampled = doc_index
        last_iter_entropy = doc_avg_system_entropy

  if doc_to_be_sampled is not None:
    try:
      vote = unknown_votes[doc_to_be_sampled].pop()
      known_votes[doc_to_be_sampled].append(vote)
    except IndexError:
      vote = None
      #TODO(What better can be done? I'm taking a random vote, if no vote is available)
      known_votes[doc_to_be_sampled].append(bool(random.randint(0,1)))

    labels = estimator(texts, known_votes, X, text_similarity, *args)  
    known_votes = add_lambda_votes_to_vote_lists(known_votes, labels)

  try:
    estimates = final_estimator(texts, known_votes, X, text_similarity, *args)
    #sys.stderr.write('Success\n')
    return get_accuracy(estimates, truths)
  except Exception, e:
    traceback.print_exc()
    #sys.stdout.write('Fail\n')
    return None    

def get_last_accuracy_in_sequence(estimator, n_votes_to_sample, texts, 
  vote_lists, truths, X, text_similarity, idx=None, return_final=False, *args):
  """ Randomly sample votes and re-calculate estimates.
  """
  random.seed()

  unknown_votes = copy_and_shuffle_sublists(vote_lists)
  known_votes = [ [] for _ in unknown_votes ]

  estimates = [None for _ in vote_lists]

  accuracy_sequence = [None] * n_votes_to_sample

  # This is a crowdsourcing procedure
  for index in xrange(n_votes_to_sample):
    # Counter
    # sys.stderr.write(str(index)+'\n')

    # Draw one vote for a random document
    updated_doc_idx = random.randrange(len(vote_lists))
    if not unknown_votes[updated_doc_idx]:
      # We ran out of votes for this document, diregard this sequence
      return None
    vote = unknown_votes[updated_doc_idx].pop()
    known_votes[updated_doc_idx].append(vote)
    
  # Calculate all the estimates
  try:
    estimates = estimator(texts, known_votes, X, text_similarity, *args)
    #sys.stderr.write('Success\n')
    return get_accuracy(estimates, truths)
  except Exception, e:
    traceback.print_exc()
    #sys.stdout.write('Fail\n')
    return None
    

def print_accuracies_to_stderr(estimator_dict, max_votes_per_doc, topic_id, sampler=get_last_accuracy_in_sequence, final_estimator=None ):
  texts, vote_lists, truths = texts_vote_lists_truths_by_topic_id[topic_id]
  n_documents = len(texts)

  pickle_file = open('../data/vectors.pkl', 'rb')
  vectors = pickle.load(pickle_file)
  X = sparse.csr_matrix(numpy.array(vectors[topic_id]).astype(numpy.double))

  text_similarity = cosine_similarity(X)

  sequence_length = int(max_votes_per_doc * n_documents)

  for estimator_name, estimator_args in estimator_dict.iteritems():
    estimator, args = estimator_args
    if sampler == get_last_accuracy_in_sequence:
      accuracy = sampler(estimator, sequence_length, texts,
          vote_lists, truths, X, text_similarity, None, False, *args)  
      accuracy_to_str = lambda acc: ("%.4f" % acc) if acc is not None else 'NA' 
      sys.stderr.write("%s\t%s\t%s\n" % ( estimator_name, topic_id,
        accuracy_to_str(accuracy) ))
    elif sampler == sample_and_minimise_entropy:
      accuracy = sampler(final_estimator, estimator, sequence_length, texts,
          vote_lists, truths, X, text_similarity, None, False, *args)  
      accuracy_to_str = lambda acc: ("%.4f" % acc) if acc is not None else 'NA' 
      sys.stderr.write("%s\t%s\t%s\n" % ( estimator_name, topic_id,
        accuracy_to_str(accuracy) ))



def print_final_accuracy_to_stream(estimator, args, topic_id, stream):
  texts, vote_lists, truths = texts_vote_lists_truths_by_topic_id[topic_id]
  n_documents = len(texts)

  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(texts)
  text_similarity = cosine_similarity(X)

  try:
    estimates = estimator(texts, vote_lists, X, text_similarity, *args)
    stream.write('ACCURACY:\t%s\n' % get_accuracy(estimates, truths))
  except Exception, e:
    traceback.print_exc()

if __name__ == "__main__":
  try:
    topic_id = sys.argv[1]
  except KeyError:
    raise Error("Please suppy topic_id to get accuracies for")

  N_REPEATS = 1
  
  for _ in xrange(N_REPEATS):
    print_accuracies_to_stderr({
       #'Matlab GP' : (est_gp, [None]),
       #'MajorityVote' : (est_majority_vote, []),
       #'MergeEnoughVotes(1)' : (est_merge_enough_votes, [ 1 ]),
       #'MajorityVoteWithNN(0.5)' : (est_majority_vote_with_nn, [ 0.5 ]),
       #'ActiveMergeEnoughVotes(0.2)' : (est_active_merge_enough_votes, [0.2]),
       #'ActiveMergeEnoughVotes(0.1)' : (est_active_merge_enough_votes, [0.1]),
       #'MinimiseEntropy': (est_minimise_entropy, [est_merge_enough_votes, [1]]),
      }, 1, topic_id)
    print_accuracies_to_stderr({
       'ActiveLearning' : (est_gp_more_confidence, [None]),
      }, 1, topic_id, sampler=sample_and_minimise_entropy, final_estimator = est_gp)

