# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import tensorflow as tf
# import coref_model as cm
# import util
import jieba
import numpy as np
import gensim
from bnlp.utils import text_tool, cal_tool
from bnlp.config import bnlp_config
from bnlp.model.coreference.coref_model import CorefModel


def get_lm_emb(sentences, wv, emb_size=300, emb_layers=3):
    max_word_count = max(len(s) for s in sentences)

    lm_emb = list()
    for sent in sentences:
        sent_emb = [wv[word] if word in wv else np.array([0.0]*emb_size) for word in sent]

        while len(sent_emb) < max_word_count:
            sent_emb.append(np.array([0.0]*emb_size))

        lm_emb.append(sent_emb)

    lm_emb = np.array(lm_emb)
    lm_emb = np.stack([lm_emb]*emb_layers, axis=-1)

    return lm_emb


def create_example(text, wv, conf):
    raw_sentences = text_tool.sentences_seg(text)
    sentences = [jieba.lcut(s, HMM=False) for s in raw_sentences]

    speakers = [["" for _ in sentence] for sentence in sentences]

    sentences_lm_emb = get_lm_emb(sentences, wv, conf["lm_size"], conf["lm_layers"])

    return {
      "doc_key": "nw",
      "clusters": [],
      "sentences": sentences,
      "speakers": speakers,
      "sentences_lm_emb": sentences_lm_emb
    }


def print_predictions(example):
    words = cal_tool.flatten(example["sentences"])
    for cluster in example["predicted_clusters"]:
        print(u"Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))


def make_predictions(text, model, wv, conf):
    example = create_example(text, wv, conf)

    tensorized_example = model.tensorize_example(example, is_training=False)
    feed_dict = {i: t for i, t in zip(model.input_tensors, tensorized_example)}
    _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores \
        = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)

    predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)
    example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
    example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
    example["head_scores"] = head_scores.tolist()
    return example


if __name__ == "__main__":
    name = sys.argv[1]
    print("Running experiment: {}".format(name))
    config = bnlp_config.CoreferenceConfigLoader(name=name).conf
    print("Loading word2vec...")
    wv = gensim.models.KeyedVectors.load_word2vec_format(config["w2v_embedding"]["path"], binary=False, unicode_errors='ignore')
    print("="*20)
    model = CorefModel(config)
    with tf.Session() as session:
        model.restore(session)
        while True:
            text = input("Document text: ")
            if text.strip():
                print_predictions(make_predictions(text, model, wv, config))

