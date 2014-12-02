from data import texts_vote_lists_truths_by_topic_id
import numpy as np
from itertools import izip, ifilter, chain, imap
import random
from plots import plot_learning_curve, plot_lines
from scipy.stats import nanmean
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

class PrintCounter(object):
  def __init__(self, count_to):
    self.count_to = count_to
    self.count = 0

  def __call__(self):
    self.count += 1
    if self.count <= self.count_to:
      print '%s / %s' % (self.count, self.count_to)


def get_accuracy(estimates, truths):
  """ 
  This gets boolean lists of estimates and truths with corresponding
  positions and returns a fraction of matching items

  If any of the pair (estimate, truth) is None, it is disregarded

  Symmetric w.r.t. argument order
  """
  pairs = izip(estimates, truths)
  pairs_without_Nones = ifilter(lambda x: None not in x, pairs)
  matching = [x == y for (x, y) in pairs_without_Nones]
  if not matching:
    return None
  else:
    return np.mean(matching)


unit_to_bool_indecisive = lambda x: None if x == 0.5 else (x > 0.5)


get_mean_vote = lambda vote_list: np.mean(vote_list) if vote_list else None


def p_majority_vote(texts, vote_lists):
  """ This is how all confidence functions should look like
      Return value in [0, 1] means certainty in document's relevance
  """
  return imap(get_mean_vote, vote_lists)


def est_majority_vote(texts, vote_lists):
  """ This is how all estimator functions should look like
  """
  return ( unit_to_bool_indecisive(conf) for conf in p_majority_vote(texts, vote_lists) )


def copy_and_shuffle_sublists(list_of_lists):
  """ Get a copy with all lists shuffled
  Use this to draw 'random' votes with .pop()
  """
  return [sorted(l, key=lambda x: random.random()) for l in list_of_lists]


def get_accuracy_sequence(estimator, n_votes_to_sample, texts, 
  vote_lists, truths, idx=None, *args):
  """ Randomly sample votes and re-calculate estimates
  """
  if idx:
    print idx

  unknown_votes = copy_and_shuffle_sublists(vote_lists)
  known_votes = [ [] for _ in unknown_votes ]

  estimates = [None for _ in vote_lists]

  accuracy_sequence = [None] * n_votes_to_sample

  for index in xrange(n_votes_to_sample):
    # Draw one vote for a random document
    updated_doc_idx = random.randrange(len(vote_lists))
    if not unknown_votes[updated_doc_idx]:
      # We ran out of votes for this document, diregard this sequence
      return None
    vote = unknown_votes[updated_doc_idx].pop()
    known_votes[updated_doc_idx].append(vote)
    
    # Recalculate all the estimates for the sake of consistency
    estimates = estimator(texts, known_votes, *args)

    # Calucate the accuracy_sequence
    accuracy_sequence[index] = get_accuracy(estimates, truths)

  return accuracy_sequence


def index_sublist_items(list_of_lists):
  """
  >>> a = [[1, 2], [65, 66], [12, 13, 14]]
  >>> list(index_sublist_items(a))
  [(0, 1), (0, 2), (1, 65), (1, 66), (2, 12), (2, 13), (2, 14)]
  """
  indexed_items = [ [ (idx, list_el) for list_el in l ]
    for idx, l in enumerate(list_of_lists) ]
  return list(chain(*indexed_items))


def get_accuracy_sequence_sample_votes(estimator, n_votes_to_sample,
  texts, vote_lists, truths):
  """ Sample random (document, vote) pairs instead of getting votes 
      for random document
  """
  doc_vote_pairs = index_sublist_items(vote_lists)
  pass


def plot_discrete_accuracies(topic_id, n_runs):
  # Building stuff from scratch, to then substitue with known functions and debug
  # Sample 1 vote per every document, then 2 votes,...
  texts, vote_lists, truths = texts_vote_lists_truths_by_topic_id[topic_id]

  all_data_estimate = est_majority_vote(texts, vote_lists)
  all_data_estimate_accuracy = get_accuracy(all_data_estimate, truths)
  print 'all data estimate accuracy: %s' % all_data_estimate_accuracy

  # minimum votes per document in this topic
  min_votes_per_doc = min([len(votes) for votes in vote_lists])
  votes_per_doc_seq = range(1, min_votes_per_doc + 1)

  accuracies_accross_runs = np.zeros( (n_runs, min_votes_per_doc) )
  final_accuracies = np.zeros(n_runs)
  for i in xrange(n_runs):
    estimates = [None] * len(texts)
    unknown_votes = copy_and_shuffle_sublists(vote_lists)
    known_votes = [ [] for _ in unknown_votes ]

    accuracies = []
    for votes_per_doc in votes_per_doc_seq:
      # draw one more vote for every document
      for doc_idx, _ in enumerate(texts):
        known_votes[doc_idx].append(unknown_votes[doc_idx].pop())

      # calculate accuracy
      estimate = est_majority_vote(texts, known_votes)
      accuracies.append( get_accuracy(estimate, truths) )

    accuracies_accross_runs[i, :] = accuracies

    # draw all the residual votes
    for doc_idx, _ in enumerate(texts):
      known_votes[doc_idx] += unknown_votes[doc_idx]

    # calculate final accuracy
    estimate = est_majority_vote(texts, known_votes)
    final_accuracies[i] = get_accuracy(estimate, truths)

  mean_accuracies = np.mean(accuracies_accross_runs, axis=0)
  plot_lines('Topic %s: accuracies at k votes per doc, %s runs' % (topic_id, n_runs),
   votes_per_doc_seq, mean_accuracies, 'Votes per document', 'Mean accuracy',
   baseline=np.mean(final_accuracies))


def plot_learning_curves_for_topic(topic_id, n_runs, votes_per_doc, estimators_dict, comment=None):
  texts, vote_lists, truths = texts_vote_lists_truths_by_topic_id[topic_id]
  n_documents = len(texts)

  min_votes_per_doc, max_votes_per_doc = votes_per_doc
  start_idx, stop_idx = min_votes_per_doc * n_documents, max_votes_per_doc * n_documents
  x = np.arange(float(start_idx), float(stop_idx)) / n_documents

  estimator_y = {}

  for estimator_name, estimator_and_args in estimators_dict.iteritems():
    print 'Calculating for %s' % estimator_name
    estimator, args = estimator_and_args
    sequences = Parallel(n_jobs=4)( delayed(get_accuracy_sequence)(estimator, stop_idx, texts, 
        vote_lists, truths, idx, *args) for idx in xrange(n_runs) )

    good_slices = [ s[start_idx:] for s in sequences if s is not None ]
    results = np.vstack(good_slices)

    estimator_y[estimator_name] = np.mean(results, axis=0)

  if comment:
    title = 'Topic %s, %s runs, %s' % (topic_id, n_runs, comment)
  else:
    title = 'Topic %s, %s runs' % (topic_id, n_runs)
  plot_learning_curve(title, x, estimator_y, 'Votes per document', 'Accuracy')


def get_p_and_var(vote_list):
  if not vote_list:
    return None, None

  p = get_mean_vote(vote_list)
  if p is None:
    return None, None
  n = len(vote_list)

  # Variance is None if there is only one vote
  var = p * (1 - p) / n if n > 1 else None
  return p, var


def is_doc_variance_better(doc_var, neighbor_var):
  """ Returns True if the document variance is less than leighbor variance
  """
  if neighbor_var is None:
    return True
  else:
    if doc_var is None:
      return False
    else:
      return (doc_var < neighbor_var)


def p_majority_vote_or_nn(texts, vote_lists, sufficient_similarity=0.5):
  """ If the nearest neighbor's similarity to you is bigger than sufficient_similarity
      and variance smaller than yours, take neighbor's conf instead of yours
  """
  vectorizer = TfidfVectorizer()
  tfidf = vectorizer.fit_transform(texts)
  similarity = cosine_similarity(tfidf)

  result_p = []
  for doc_index, vote_list in enumerate(vote_lists):
    doc_p, doc_var = get_p_and_var(vote_list)
    similarities = similarity[:, doc_index]
    similarities[doc_index] = 0
    nn_similarity = similarities.max()

    if nn_similarity > sufficient_similarity:
      nn_index = similarities.argmax()
      nn_p, nn_var = get_p_and_var(vote_lists[nn_index])
      p = doc_p if is_doc_variance_better(doc_var, nn_var) else nn_p
    else:
      p = doc_p
    result_p.append(p)

  return result_p


def est_majority_vote_or_nn(texts, vote_lists, sufficient_similarity=0.5):
  return ( unit_to_bool_indecisive(conf) for conf
   in p_majority_vote_or_nn(texts, vote_lists, sufficient_similarity) )


print "plotting curves from 1 to 7 votes per doc"
print "started job at %s" % datetime.datetime.now()
plot_learning_curves_for_topic('20690', 10, (1,5), { 
#  'Majority vote' : est_majority_vote,
  'Majority vote or NN, suff.sim. 0.2': (est_majority_vote_or_nn, [ 0.2 ]),
  'Majority vote or NN, suff.sim. 0.5': (est_majority_vote_or_nn, [ 0.5 ]),
})
print "finished job at %s" % datetime.datetime.now()
