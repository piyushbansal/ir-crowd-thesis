from experiments import get_accuracy, est_gp, est_majority_vote, copy_and_shuffle_sublists, est_merge_enough_votes, est_majority_vote_with_nn
from data import texts_vote_lists_truths_by_topic_id
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import codecs
import sys
import random
import traceback

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
    

def print_accuracies_to_stderr(estimator_dict, max_votes_per_doc, topic_id):
  texts, vote_lists, truths = texts_vote_lists_truths_by_topic_id[topic_id]
  n_documents = len(texts)

  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(texts)
  text_similarity = cosine_similarity(X)

  sequence_length = int(max_votes_per_doc * n_documents)

  for estimator_name, estimator_args in estimator_dict.iteritems():
    estimator, args = estimator_args
    accuracy = get_last_accuracy_in_sequence(estimator, sequence_length, texts,
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

  N_REPEATS = 10
  
  for _ in xrange(N_REPEATS):
    print_accuracies_to_stderr({
       'Matlab GP' : (est_gp, [None]),
       'MajorityVote' : (est_majority_vote, []),
       'MergeEnoughVotes(1)' : (est_merge_enough_votes, [ 1 ]),
       'MajorityVoteWithNN(0.5)' : (est_majority_vote_with_nn, [ 0.5 ]),
      }, 1, topic_id)

