"""
Training the SNRM model.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""
import pymongo
import logging
import numpy as np
import tensorflow as tf
import json
from dictionary import Dictionary
from params import FLAGS
from snrm import SNRM
import time
from util import check_gpu_available, my_tokenize
import random


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

# layer_size is a list containing the size of each layer. It can be set through the 'hiddein_x' arguments.
layer_size = [FLAGS.emb_dim]
for i in [FLAGS.hidden_1, FLAGS.hidden_2, FLAGS.hidden_3]:  # FLAGS.hidden_4, FLAGS.hidden_5]:
    if i <= 0:
        break
    layer_size.append(i)

# Dictionary is a class containing terms and their IDs. The implemented class just load the terms from a Galago dump
# file. If you are not using Galago, you have to implement your own reader. See the 'dictionary.py' file.
dictionary = Dictionary()
print("loading dictionary...")
# dictionary.load_from_galago_dump(FLAGS.base_path + FLAGS.dict_file_name, FLAGS.dict_min_freq)
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


client = pymongo.MongoClient()
db = client.snrm
# doc_coll = db.docs_exp
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


def generate_batch(batch_size, mode='train'):
    """
        Generating pairwise training or validation data for each batch. This function should be implemented.
        Note: For unknown terms term ID should be set to zero. Please use the dictionary size for padding. In other
        words, padding value should be |V|+1, where |V| is the vocabulary size.
        Args:
            batch_size (int): total number of training or validation data in each batch.
            mode (str): should be either 'train' or 'valid'.

        Returns:
            batch_query (list): a 2D list of int containing query term IDs with size (batch_size * FLAGS.max_q_len).
            batch_doc1 (list): a 2D list of int containing doc 1 term IDs with size (batch_size * FLAGS.max_doc_len).
            batch_doc2 (list): a 2D list of int containing doc 2 term IDs with size (batch_size * FLAGS.max_doc_len).
            batch_label (list): a 2D list of float within the range of [0, 1] with size (batch_size * 1).
             Label shows the probability of doc1 being more relevant than doc2. This can simply be 0 or 1.
    """
    # raise Exception('the generate_batch method is not implemented.')
    global batch_index

    batch_query = []
    batch_doc1 = []
    batch_doc2 = []
    batch_label = []

    if batch_index+batch_size >= data_size:
        batch_index = batch_index + batch_size - data_size
    batch_data = pair_wise_data[batch_index:batch_index+batch_size]
    random.shuffle(batch_data)  # shuffle the order

    for data in batch_data:
        q = tokens2vec(my_tokenize(data["q"]), FLAGS.max_q_len)
        d1 = tokens2vec(get_tokens(data["d1_id"]), FLAGS.max_doc_len)
        d2 = tokens2vec(get_tokens(data["d2_id"]), FLAGS.max_doc_len)
        batch_query.append(q)
        batch_doc1.append(d1)
        batch_doc2.append(d2)
        if data["label"] == -1:
            batch_label.append(0.0)
        else:
            batch_label.append(1.0)

    batch_index += batch_size

    return batch_query, batch_doc1, batch_doc2, batch_label


batch_index = 0
print("loading pair wise dataset")
with open("../data/pair_wise_data.json", "r") as f:
    pair_wise_data = json.load(f)
data_size = len(pair_wise_data)
print("dataset loaded")

# # check the gpu availability
# while not check_gpu_available():
#     time.sleep(1)

writer = tf.summary.FileWriter(FLAGS.base_path + FLAGS.log_path + FLAGS.run_name, graph=snrm.graph)

# Launch the graph
with tf.Session(graph=snrm.graph) as session:
    session.run(snrm.init)
    logging.info('Initialized')

    ckpt = tf.train.get_checkpoint_state(FLAGS.base_path + FLAGS.model_path + FLAGS.run_name)

    if ckpt and ckpt.model_checkpoint_path:
        logging.info(ckpt.model_checkpoint_path)
        snrm.saver.restore(session, ckpt.model_checkpoint_path)  # restore all variables
        logging.info('Load model from {:s}'.format(ckpt.model_checkpoint_path))

    # training
    if not FLAGS.experiment_mode:
        num_steps = FLAGS.num_train_steps
        average_loss = 0
        for step in range(num_steps):
            query, doc1, doc2, labels = generate_batch(FLAGS.batch_size, 'train')
            labels = np.array(labels)
            labels = np.concatenate(
                [labels.reshape(FLAGS.batch_size, 1), 1.-labels.reshape(FLAGS.batch_size, 1)], axis=1)
            feed_dict = {snrm.query_pl: query,
                         snrm.doc1_pl: doc1,
                         snrm.doc2_pl: doc2,
                         snrm.labels_pl: labels}
            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val, summary = session.run([snrm.optimizer, snrm.loss, snrm.summary_op], feed_dict=feed_dict)

            writer.add_summary(summary, step)
            if step % 10 == 0:
                print(step, batch_index, time.strftime("%Y-%m-%d %H:%M:%S"))

            if (step % FLAGS.validate_every_n_steps == 0) and (step != 0):
                valid_coss = 0.
                valid_id = 0
                doc_total_len = 0
                doc_count = 0
                q_total_len = 0
                q_count = 0

                all_zero_d_count = 0
                all_zero_q_count = 0
                for valid_step in range(FLAGS.num_valid_steps):
                    query, doc1, doc2, labels = generate_batch(FLAGS.batch_size, 'valid')
                    labels = np.array(labels)
                    labels = np.concatenate(
                        [labels.reshape(FLAGS.batch_size, 1), 1. - labels.reshape(FLAGS.batch_size, 1)], axis=1)
                    # print(labels)
                    feed_dict = {snrm.query_pl: query,
                                 snrm.doc1_pl: doc1,
                                 snrm.doc2_pl: doc2,
                                 snrm.labels_pl: labels}
                    cost_val, doc_repr1, q_repr = session.run([snrm.cost,
                                                               snrm.d1_repr,
                                                               snrm.q_repr],
                                                              feed_dict=feed_dict)

                    # doc_repr1 = session.run(snrm.doc_representation, feed_dict={snrm.doc_pl: doc1})
                    # q_repr = session.run(snrm.query_representation, feed_dict={snrm.query_pl: query})

                    for i in range(FLAGS.batch_size):
                        d_c = 0  # 本次的document not zero count
                        for j in range(len(doc_repr1[i])):
                            if doc_repr1[i][j] > 0.:
                                d_c += 1
                        if d_c == 0:
                            all_zero_d_count += 1
                        doc_total_len += d_c
                    doc_count += FLAGS.batch_size

                    for i in range(FLAGS.batch_size):
                        q_c = 0  # 本次的query not zero count
                        for j in range(len(q_repr[i])):
                            if q_repr[i][j] > 0.:
                                q_c += 1
                        if q_c == 0:
                            all_zero_q_count += 1
                        q_total_len += q_c
                    q_count += FLAGS.batch_size
                    valid_coss += cost_val

                valid_coss /= FLAGS.num_valid_steps
                print('Average cost on validation set at step ', step, ': ', valid_coss)
                print('Doc Avg Length at step ', step, ': ', doc_total_len / doc_count,
                      ", zc-->", all_zero_d_count)
                print('Query Avg Length at step ', step, ': ', q_total_len / q_count,
                      ", zc-->", all_zero_d_count)

            if step > 0 and step % FLAGS.save_snapshot_every_n_steps == 0:
                save_path = snrm.saver.save(session, FLAGS.base_path + FLAGS.model_path + FLAGS.run_name + str(step))
                print('Model saved in file: %s' % save_path)

        save_path = snrm.saver.save(session, FLAGS.base_path + FLAGS.model_path + FLAGS.run_name)
        print('Model saved in file: %s' % save_path)

    else:
        print('Experiment Mode is ON!')
        # inference should be done. You should implement it. It's easy. Please refer to the paper. You should just
        # construct the inverted index from the learned representations. Then the query should fed to the network and
        # the documents that contain the "query latent terms" should be scored and ranked. If you have any question,
        # please do not hesitate to contact me (zamani@cs.umass.edu).
