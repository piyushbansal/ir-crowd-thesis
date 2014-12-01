from data import texts_vote_lists_truths_by_topic_id
import numpy as np
from itertools import izip, ifilter, chain, imap
import random
from plots import plot_learning_curve, plot_lines
from scipy.stats import nanmean
from sklearn.externals.joblib import Parallel, delayed

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


def get_accuracy_sequence(estimator, n_votes_to_sample, texts, vote_lists, truths, idx=None):
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
    estimates = estimator(texts, known_votes)

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


def plot_learning_curve_for_topic(topic_id, n_runs, votes_per_doc=(1,10)):
  data = texts_vote_lists_truths_by_topic_id[topic_id]
  estimator = est_majority_vote

  texts, vote_lists, truths = data
  n_documents = len(texts)

  estimates = estimator(texts, vote_lists)

  max_accuracy = get_accuracy(estimates, truths)
  print 'max accuracy %s' % max_accuracy

  min_votes_per_doc, max_votes_per_doc = votes_per_doc
  start_idx, stop_idx = min_votes_per_doc * n_documents, max_votes_per_doc * n_documents

  sequences = Parallel(n_jobs=4)( delayed(get_accuracy_sequence)(estimator, stop_idx, texts, 
      vote_lists, truths, idx) for idx in xrange(n_runs) )

  good_slices = [ s[start_idx:] for s in sequences if s is not None ]
  results = np.vstack(good_slices)

  x = np.arange(float(start_idx), float(stop_idx)) / n_documents
  y = np.mean(results, axis=0)

  print 'last iteration accuracy %s' % y[-1]

  plot_learning_curve('Learning curve for topic %s, %s runs' % 
    (topic_id, n_runs), x, {'majority voting' : y }, 
    'Votes per document', 'Accuracy', baseline=max_accuracy)


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


def plot_learning_curves_for_topic(topic_id, n_runs, votes_per_doc, esimators_dict):
  texts, vote_lists, truths = texts_vote_lists_truths_by_topic_id[topic_id]
  n_documents = len(texts)

  min_votes_per_doc, max_votes_per_doc = votes_per_doc
  start_idx, stop_idx = min_votes_per_doc * n_documents, max_votes_per_doc * n_documents
  x = np.arange(float(start_idx), float(stop_idx)) / n_documents

  estimator_y = {}

  for estimator_name, estimator in esimators_dict.iteritems():
    sequences = Parallel(n_jobs=4)( delayed(get_accuracy_sequence)(estimator, stop_idx, texts, 
        vote_lists, truths, idx) for idx in xrange(n_runs) )

    good_slices = [ s[start_idx:] for s in sequences if s is not None ]
    results = np.vstack(good_slices)

    estimator_y[estimator_name] = np.mean(results, axis=0)

  plot_learning_curve('Learning curve for topic %s, %s runs' % 
    (topic_id, n_runs), x, estimator_y, 
    'Votes per document', 'Accuracy')


def func_linear_combination(estimator1, estimator2, propotion=0.5):
  def result(*args, **kwargs):
    return unit_to_bool_indecisive(
      proportion * estimator1(*args, **kwargs) + (1 - proportion) * estimator2(*args, **kwargs)
    )

  return result


def get_p_and_var(vote_list):
  if not vote_list:
    return None, None

  p = get_mean_vote(vote_list)
  n = len(vote_list)

  # Variance is None if there is only one vote
  var = p * (1 - p) / n if n > 1 else None
  return p, var


def var_lr(doc_var, neighbor_var):
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
  nn_search_structure = KDTree(similarity)

  result_p = []
  for doc_index, vote_list in enumerate(vote_lists):
    p_doc, var_doc = get_p_and_var(vote_list)
    # TODO Find out nearest neighbor

plot_learning_curves_for_topic('20932', 100, (1,12), { 'Majority vote' : est_majority_vote })