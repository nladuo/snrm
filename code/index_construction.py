"""
Inverted index construction from the latent terms to document IDs from the representations learned by SNRM.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

import logging
import tensorflow as tf
import pymongo
import traceback
import numpy as np
from dictionary import Dictionary
from inverted_index import InMemoryInvertedIndex
from params import FLAGS
from snrm import SNRM

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


client = pymongo.MongoClient()
db = client.snrm
doc_coll = db.docs


def tokens2vec(tokens, length):
    data = [0] * length
    count = 0
    for token in tokens:
        if token in dictionary.term_to_id.keys():
            data[count] = dictionary.term_to_id[token]
            count += 1
    return data


def get_tokens(docNo):
    return doc_coll.find_one({"docNo": docNo})["tokens"]


def generate_batch(batch_size):
    """
        Generating a batch of documents from the collection for making the inverted index. This function should iterate
        over all the documents (each once) to learn an inverted index.
        Args:
            batch_size (int): total number of training or validation data in each batch.

        Returns:
            batch_doc_id (list): a list of str containing document IDs.
            batch_doc (list): a 2D list of int containing document term IDs with size (batch_size * FLAGS.max_doc_len).
    """
    # raise Exception('the generate_batch method is not implemented.')
    global batch_index

    batch_doc_id = document_ids[batch_index:batch_index+batch_size]
    batch_doc = []
    for doc_id in batch_doc_id:
        d = tokens2vec(get_tokens(doc_id), FLAGS.max_doc_len)
        batch_doc.append(d)

    batch_index += batch_size

    return batch_doc_id, batch_doc


batch_index = 0
document_ids = []
print("loading document_ids...")
for doc in doc_coll.find({}):
    token_len = len(doc["tokens"])
    if token_len == 0 or token_len > 4000:
        continue
    document_ids.append(doc["docNo"])

print("doc length:", len(document_ids))

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

inverted_index = InMemoryInvertedIndex(layer_size[-1])
with tf.Session(graph=snrm.graph) as session:
    session.run(snrm.init)
    print('Initialized')

    c = "32000"  # my trained "model/nladuo-snrm32000.data-00000-of-00001"
    snrm.saver.restore(session, FLAGS.base_path + FLAGS.model_path +
                       FLAGS.run_name + model_index)  # restore all variables
    logging.info('Load model from {:s}'.format(FLAGS.base_path + FLAGS.model_path + FLAGS.run_name + model_index))

    docs = []
    doc_names = []
    col_file = open(FLAGS.base_path + FLAGS.model_path + 'learned_robust.txt', 'wb')
    doc_len_file = open(FLAGS.base_path + FLAGS.model_path + 'learned_robust_doc_len.txt', 'wb')

    while True:
        print("indexing", batch_index, "to", batch_index+FLAGS.batch_size, "...")
        doc_ids, docs = generate_batch(FLAGS.batch_size)
        try:
            doc_repr = session.run(snrm.doc_representation, feed_dict={snrm.doc_pl: docs})
            inverted_index.add(doc_ids, doc_repr)
        except Exception as ex:
            traceback.print_exc()
            break

    inverted_index.store(FLAGS.base_path + FLAGS.model_path + FLAGS.run_name + '-inverted-index.pkl')
