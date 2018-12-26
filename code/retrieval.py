"""
The inference (retrieval) sample file.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

import logging
import tensorflow as tf
import pymongo
from dictionary import Dictionary
from inverted_index import InMemoryInvertedIndex
from params import FLAGS
from snrm import SNRM
import pickle as pkl

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

# layer_size is a list containing the size of each layer. It can be set through the 'hiddein_x' arguments.
layer_size = [FLAGS.emb_dim]
for i in [FLAGS.hidden_1, FLAGS.hidden_2, FLAGS.hidden_3, FLAGS.hidden_4, FLAGS.hidden_5]:
    if i <= 0:
        break
    layer_size.append(i)

# Dictionary is a class containing terms and their IDs. The implemented class just load the terms from a Galago dump
# file. If you are not using Galago, you have to implement your own reader. See the 'dictionary.py' file.
dictionary = Dictionary()
# dictionary.load_from_galago_dump(FLAGS.base_path + FLAGS.dict_file_name, FLAGS.dict_min_freq)
print("loading dictionary...")
dictionary.load_my_dict(FLAGS.base_path + FLAGS.dict_file_name, FLAGS.dict_min_freq)

print("creating SNRM model...")
# The SNRM model.
snrm = SNRM(dictionary=dictionary,
            pre_trained_embedding_file_name=FLAGS.base_path + FLAGS.pre_trained_embedding_file_name,
            batch_size=FLAGS.batch_size,
            max_q_len=FLAGS.max_q_len,
            max_doc_len=FLAGS.max_doc_len,
            emb_dim=FLAGS.emb_dim,
            layer_size=layer_size,
            dropout_parameter=FLAGS.dropout_parameter,
            regularization_term=FLAGS.regularization_term,
            learning_rate=FLAGS.learning_rate)

print("loading inverted-index...")
inverted_index = InMemoryInvertedIndex(layer_size[-1])
inverted_index.load(FLAGS.base_path + FLAGS.model_path + FLAGS.run_name + '-inverted-index.pkl')


client = pymongo.MongoClient()
db = client.snrm
query_coll = db.queries

with tf.Session(graph=snrm.graph) as session:
    session.run(snrm.init)
    print('Initialized')

    model_index = "32000"  # my trained "model/nladuo-snrm32000.data-00000-of-00001"
    snrm.saver.restore(session, FLAGS.base_path + FLAGS.model_path +
                       FLAGS.run_name + model_index)  # restore all variables
    logging.info('Load model from {:s}'.format(FLAGS.base_path + FLAGS.model_path + FLAGS.run_name + model_index))

    queries = {}
    for q in query_coll.find({}):
        qid = str(q["number"])
        queries[qid] = q["title"].lower()

    result = dict()
    for qid in queries.keys():
        logging.info('processing query #' + qid + ': ' + queries[qid])
        q_term_ids = dictionary.get_emb_list(queries[qid], delimiter=' ')

        for i in range(len(q_term_ids), FLAGS.max_q_len):
            q_term_ids.append(0)

        query_repr = session.run(snrm.query_representation, feed_dict={snrm.test_query_pl: [q_term_ids]})
        retrieval_scores = dict()
        for i in range(len(query_repr[0])):
            if query_repr[0][i] > 0.:
                for (did, weight) in inverted_index.index[i]:
                    if did not in retrieval_scores:
                        retrieval_scores[did] = 0.
                    retrieval_scores[did] += query_repr[0][i] * weight

        result[qid] = sorted(retrieval_scores.items(), key=lambda x: x[1])
    pkl.dump(result, open(FLAGS.base_path + FLAGS.result_path + FLAGS.run_name + '-250-queries.pkl', "wb"))
