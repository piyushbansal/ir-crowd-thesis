from data import texts_vote_lists_truths_by_topic_id
from gensim.models import Doc2Vec
import nltk
from nltk.corpus import stopwords

import pickle

DOC2VEC_MODEL = '/home/pbansal/CrowdSourcingMinimal/data/doc2vec2.model'

def get_model():
  try:
    model = Doc2Vec.load(DOC2VEC_MODEL)
    return model
  except:
    print "Model couldn't be loaded"
    return None

def get_vectors():
  """Gets Doc2Vec vectors of the texts in corpus, and returns a dictioanry of the form
     TopicId: (Doc1Vector, DocVector2 ... for all the docs in this topicID)
  """
  model = get_model()
  doc_vectors_by_topic_id = {}
  stopword_list = set(stopwords.words('english'))
  pickle_file = open('vectors.pkl', 'wb')

  if model:
    for topic_id in texts_vote_lists_truths_by_topic_id:
      texts, vote_lists, truths = texts_vote_lists_truths_by_topic_id[topic_id]
      
      doc_vectors_for_current_topic = []
      for text in texts:
        words = [word for word in text.split() if word not in stopword_list]
        print "WORDS:", words
        doc2vec_vector = model.infer_vector(words)
        print "DOC2VEC_VECTOR:", doc2vec_vector
        doc_vectors_for_current_topic.append(doc2vec_vector)
      
      doc_vectors_by_topic_id[topic_id] = doc_vectors_for_current_topic
    pickle.dump(doc_vectors_by_topic_id, pickle_file)
    return doc_vectors_by_topic_id


if __name__ == "__main__":
    get_vectors() 
 
        
        
        
      
      

