# -*- coding: utf-8 -*-
"""
    Created by JoeYip on 29/03/2019

    :copyright: (c) 2019 by nesting.xyz
    :license: BSD, see LICENSE for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import re
import subprocess
import sys
import os
import tempfile
import json
import collections

import gensim
import h5py
import numpy as np

from bnlp import project_dir
from bnlp.config.bnlp_config import w2v_config_loader
from bnlp.data.const import DataSetType
from bnlp.utils import cal_tool

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")
COREF_RESULTS_REGEX = re.compile(
    r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*",
    re.DOTALL)


class DocumentState(object):
    def __init__(self):
        self.doc_key = None
        self.text = []
        self.text_speakers = []
        self.speakers = []
        self.sentences = []
        self.constituents = {}
        self.const_stack = []
        self.ner = {}
        self.ner_stack = []
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)

    def assert_empty(self):
        assert self.doc_key is None
        assert len(self.text) == 0
        assert len(self.text_speakers) == 0
        assert len(self.speakers) == 0
        assert len(self.sentences) == 0
        assert len(self.constituents) == 0
        assert len(self.const_stack) == 0
        assert len(self.ner) == 0
        assert len(self.ner_stack) == 0
        assert len(self.coref_stacks) == 0
        assert len(self.clusters) == 0

    def assert_finalizable(self):
        assert self.doc_key is not None
        assert len(self.text) == 0
        assert len(self.text_speakers) == 0
        assert len(self.speakers) > 0
        assert len(self.sentences) > 0
        assert len(self.constituents) > 0
        assert len(self.const_stack) == 0
        assert len(self.ner_stack) == 0
        assert all(len(s) == 0 for s in self.coref_stacks.values())

    def span_dict_to_list(self, span_dict):
        return [(s, e, l) for (s, e), l in span_dict.items()]

    def finalize(self):
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        merged_clusters = [list(c) for c in merged_clusters]
        all_mentions = cal_tool.flatten(merged_clusters)
        assert len(all_mentions) == len(set(all_mentions))

        return {
            "doc_key": self.doc_key,
            "sentences": self.sentences,
            "speakers": self.speakers,
            "constituents": self.span_dict_to_list(self.constituents),
            "ner": self.span_dict_to_list(self.ner),
            "clusters": merged_clusters
        }


def __get_doc_key(doc_id, part):
    return "{}_{}".format(doc_id, int(part))


def __normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def __handle_bit(word_index, bit, stack, spans):
    asterisk_idx = bit.find("*")
    if asterisk_idx >= 0:
        open_parens = bit[:asterisk_idx]
        close_parens = bit[asterisk_idx + 1:]
    else:
        open_parens = bit[:-1]
        close_parens = bit[-1]

    current_idx = open_parens.find("(")
    while current_idx >= 0:
        next_idx = open_parens.find("(", current_idx + 1)
        if next_idx >= 0:
            label = open_parens[current_idx + 1:next_idx]
        else:
            label = open_parens[current_idx + 1:]
        stack.append((word_index, label))
        current_idx = next_idx

    for c in close_parens:
        assert c == ")"
        open_index, label = stack.pop()
        current_span = (open_index, word_index)
        """
        if current_span in spans:
            spans[current_span] += "_" + label
        else:
            spans[current_span] = label
        """
        spans[current_span] = label


def __handle_line(line, document_state, language, labels, stats):
    begin_document_match = re.match(BEGIN_DOCUMENT_REGEX, line)
    if begin_document_match:
        document_state.assert_empty()
        document_state.doc_key = __get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
        return None
    elif line.startswith("#end document"):
        document_state.assert_finalizable()
        finalized_state = document_state.finalize()
        stats["num_clusters"] += len(finalized_state["clusters"])
        stats["num_mentions"] += sum(len(c) for c in finalized_state["clusters"])
        labels["{}_const_labels".format(language)].update(l for _, _, l in finalized_state["constituents"])
        labels["ner"].update(l for _, _, l in finalized_state["ner"])
        return finalized_state
    else:
        row = line.split()
        if len(row) == 0:
            stats["max_sent_len_{}".format(language)] = max(len(document_state.text),
                                                            stats["max_sent_len_{}".format(language)])
            stats["num_sents_{}".format(language)] += 1
            document_state.sentences.append(tuple(document_state.text))
            del document_state.text[:]
            document_state.speakers.append(tuple(document_state.text_speakers))
            del document_state.text_speakers[:]
            return None
        assert len(row) >= 12

        doc_key = __get_doc_key(row[0], row[1])
        word = __normalize_word(row[3], language)
        parse = row[5]
        speaker = row[9]
        ner = row[10]
        coref = row[-1]

        word_index = len(document_state.text) + sum(len(s) for s in document_state.sentences)
        document_state.text.append(word)
        document_state.text_speakers.append(speaker)

        __handle_bit(word_index, parse, document_state.const_stack, document_state.constituents)
        __handle_bit(word_index, ner, document_state.ner_stack, document_state.ner)

        if coref != "-":
            for segment in coref.split("|"):
                if segment[0] == "(":
                    if segment[-1] == ")":
                        cluster_id = int(segment[1:-1])
                        document_state.clusters[cluster_id].append((word_index, word_index))
                    else:
                        cluster_id = int(segment[1:])
                        document_state.coref_stacks[cluster_id].append(word_index)
                else:
                    cluster_id = int(segment[:-1])
                    start = document_state.coref_stacks[cluster_id].pop()
                    document_state.clusters[cluster_id].append((start, word_index))
        return None


def __minimize_partition(name, language, labels, stats, extension="v4_gold_conll"):
    input_path = "{}.{}.{}".format(name, language, extension)
    output_path = "{}.{}.jsonlines".format(name, language)
    count = 0
    print("Minimizing {}".format(input_path))
    with open(input_path, "r") as input_file:
        with open(output_path, "w") as output_file:
            document_state = DocumentState()
            for line in input_file.readlines():
                document = __handle_line(line, document_state, language, labels, stats)
                if document is not None:
                    output_file.write(json.dumps(document))
                    output_file.write("\n")
                    count += 1
                    document_state = DocumentState()
    print("Wrote {} documents to {}".format(count, output_path))


'''
将数据集拆分为dev、train、test集合
'''


def split(language, data_dir):
    labels = collections.defaultdict(set)
    stats = collections.defaultdict(int)

    __minimize_partition(os.path.join(data_dir, DataSetType.dev.value), language, labels, stats)
    __minimize_partition(os.path.join(data_dir, DataSetType.train.value), language, labels, stats)
    __minimize_partition(os.path.join(data_dir, DataSetType.test.value), language, labels, stats)


def __get_char_vocab(input_filenames, output_filename, data_dir):
    vocab = set()
    for filename in input_filenames:
        with open(os.path.join(data_dir, filename)) as f:
            for line in f.readlines():
                for sentence in json.loads(line)["sentences"]:
                    for word in sentence:
                        vocab.update(word)
    vocab = sorted(list(vocab))
    with open(os.path.join(data_dir, output_filename), "w") as f:
        for char in vocab:
            f.write("{}\n".format(char).encode("utf8").decode("utf8"))
    print("Wrote {} characters to {}".format(len(vocab), output_filename))


'''
获取CoNLL数据集中出现的字符
'''


def get_char_vocab(language, data_dir):
    __get_char_vocab(["{}.{}.jsonlines".format(partition, language) for partition in ("train", "dev", "test")],
                     "char_vocab.{}.txt".format(language), data_dir)


"""

过滤词向量，保留只在语料中出现的词向量

"""


def filt(language, data_dir):
    words_to_keep = set()  # 保存在语料中出现过的词
    for partition in ("train", "dev", "test"):
        json_filename = os.path.join(data_dir, "{}.{}.jsonlines".format(partition, language))
        with open(json_filename) as json_file:
            for line in json_file.readlines():
                for sentence in json.loads(line)["sentences"]:
                    words_to_keep.update(sentence)

    total_lines = 0
    kept_lines = 0
    out_filename = os.path.join(project_dir, w2v_config_loader.w2v_filter)
    with open(os.path.join(project_dir, w2v_config_loader.w2v), encoding='utf-8') as in_file:
        with open(out_filename, "w", encoding='utf-8') as out_file:
            in_file.readline()  # 跳过第一行
            for line in in_file.readlines():  # 读取向量
                total_lines += 1
                word = line.split()[0]
                if word in words_to_keep:
                    kept_lines += 1
                    out_file.write(line)  # 保留只在语料中出现的词向量

    print("Kept {} out of {} lines.".format(kept_lines, total_lines))
    print("Wrote result to {}.".format(out_filename))


def __load_word_vec(file_path):
    print("Loading word vec...")
    wv = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=False, unicode_errors='ignore')
    return wv


def __get_sentence_emb(word_list, wv, emb_size, max_len):
    word_vec_list = list()
    for word in word_list:
        if word in wv:
            word_vec = wv[word]

        else:
            word_vec = np.zeros((emb_size,))

        word_vec_list.append(word_vec)

    while len(word_vec_list) < max_len:
        word_vec_list.append(np.zeros((emb_size,)))

    return np.stack([word_vec_list, word_vec_list, word_vec_list], axis=-1)


def __do_cache(data_path, w2v, emb_size, out_file):
    with open(data_path) as in_file:
        for doc_num, line in enumerate(in_file.readlines()):  # 遍历处理每一个文档
            example = json.loads(line)
            sentences = example["sentences"]  # [[w1, w2...], [], ...]
            max_sentence_length = max(len(s) for s in sentences)

            text_len = np.array([len(s) for s in sentences])

            all_sentence_lm_emb = []
            for sentence in sentences:
                all_features_array = __get_sentence_emb(sentence, w2v, emb_size, max_sentence_length)
                all_sentence_lm_emb.append(all_features_array)

            all_sentence_lm_emb = np.stack(all_sentence_lm_emb, axis=0)

            if out_file:
                file_key = example["doc_key"].replace("/", ":")
                group = out_file.create_group(file_key)
                for i, (e, l) in enumerate(zip(all_sentence_lm_emb, text_len)):
                    e = e[:l, :, :]
                    group[str(i)] = e
            if doc_num % 10 == 0:
                print("Cached {} documents in {}".format(doc_num + 1, data_path))


def cache_zh(data_dir):
    w2v_config = w2v_config_loader.w2v_zh
    w2v = __load_word_vec(os.path.join(data_dir, "..", w2v_config['path']))

    with h5py.File(os.path.join(data_dir, "word2vec_zh_cache.hdf5"), "w") as out_file:
        __do_cache(os.path.join(data_dir, "train.chinese.jsonlines"), w2v, w2v_config['size'], out_file)
        __do_cache(os.path.join(data_dir, "dev.chinese.jsonlines"), w2v, w2v_config['size'], out_file)

    print("Cache word2vec finished.")


def prepare(language, data_dir):
    split(language, data_dir)
    get_char_vocab(language, data_dir)
    filt(language, data_dir)


def output_conll(input_file, output_file, predictions):
    prediction_map = {}
    for doc_key, clusters in predictions.items():
        start_map = collections.defaultdict(list)
        end_map = collections.defaultdict(list)
        word_map = collections.defaultdict(list)
        for cluster_id, mentions in enumerate(clusters):
            for start, end in mentions:
                if start == end:
                    word_map[start].append(cluster_id)
                else:
                    start_map[start].append((cluster_id, end))
                    end_map[end].append((cluster_id, start))
        for k, v in start_map.items():
            start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
        for k, v in end_map.items():
            end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
        prediction_map[doc_key] = (start_map, end_map, word_map)

    word_index = 0
    for line in input_file.readlines():
        row = line.split()
        if len(row) == 0:
            output_file.write("\n")
        elif row[0].startswith("#"):
            begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
            if begin_match:
                doc_key = __get_doc_key(begin_match.group(1), begin_match.group(2))
                start_map, end_map, word_map = prediction_map[doc_key]
                word_index = 0
            output_file.write(line)
            output_file.write("\n")
        else:
            assert __get_doc_key(row[0], row[1]) == doc_key
            coref_list = []
            if word_index in end_map:
                for cluster_id in end_map[word_index]:
                    coref_list.append("{})".format(cluster_id))
            if word_index in word_map:
                for cluster_id in word_map[word_index]:
                    coref_list.append("({})".format(cluster_id))
            if word_index in start_map:
                for cluster_id in start_map[word_index]:
                    coref_list.append("({}".format(cluster_id))

            if len(coref_list) == 0:
                row[-1] = "-"
            else:
                row[-1] = "|".join(coref_list)

            output_file.write("     ".join(row))
            output_file.write("\n")
            word_index += 1


def official_conll_eval(gold_path, predicted_path, metric, official_stdout=False):
    cmd = [os.path.join(project_dir, 'data', "conll-2012/scorer/v8.01/scorer.pl"), metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        print(stderr)

    if official_stdout:
        print("Official result for {}".format(metric))
        print(stdout)

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return {"r": recall, "p": precision, "f": f1}


def evaluate_conll(gold_path, predictions, official_stdout=False):
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as prediction_file:
        with open(gold_path, "r") as gold_file:
            output_conll(gold_file, prediction_file, predictions)
        print("Predicted conll file: {}".format(prediction_file.name))
    return {m: official_conll_eval(gold_file.name, prediction_file.name, m, official_stdout) for m in
            ("muc", "bcub", "ceafe")}
