# -*- coding: utf-8 -*-
"""
    Created by JoeYip on 29/03/2019

    :copyright: (c) 2019 by nesting.xyz
    :license: BSD, see LICENSE for more details.
"""

import os
import errno
import codecs
import collections


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            print('Making dir %s error' % path)
            raise
    return path


def load_char_dict(path):
    vocab = [u"<unk>"]
    with codecs.open(path, encoding="utf-8") as f:
        vocab.extend(l.strip() for l in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})

    return char_dict


def sentences_seg(text, punctuation_list=None):
    if not punctuation_list:
        punctuation_list = ['\n', '。', '！', '？', '……']

    text_sentences_list = list()
    tmp_list = list()
    for word in text:
        tmp_list.append(word)
        if word in punctuation_list:
            text_sentences_list.append(''.join(tmp_list).lower())
            tmp_list.clear()
    if tmp_list:
        text_sentences_list.append(''.join(tmp_list).lower())

    return text_sentences_list
