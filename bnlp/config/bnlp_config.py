# -*- coding: utf-8 -*-
"""
    Created by JoeYip on 29/03/2019

    :copyright: (c) 2019 by nesting.xyz
    :license: BSD, see LICENSE for more details.
"""

import os
import sys

import pyhocon

from bnlp import project_dir
from bnlp.utils.singleton import SingletonType
from bnlp.utils.text_tool import mkdirs


_CONFIG_DIR = project_dir + "/config/coreference.conf"


def _set_gpus(*gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


class CoreferenceConfigLoader(metaclass=SingletonType):
    def __init__(self, name):
        if "GPU" in os.environ:
            _set_gpus(int(os.environ["GPU"]))
        else:
            _set_gpus('')

        print("Running experiment: {}".format(name))

        self.conf = pyhocon.ConfigFactory.parse_file(_CONFIG_DIR)[name]
        self.conf['log_dir'] = mkdirs(os.path.join(project_dir, 'data', self.config["log_root"], name))

        print(pyhocon.HOCONConverter.convert(self.config, "hocon"))

    @property
    def config(self):
        return self.conf


class Word2VecConfigLoader(metaclass=SingletonType):
    def __init__(self):
        print("Loading word2Vec data")
        config = pyhocon.ConfigFactory.parse_file(_CONFIG_DIR)
        self.w2v_zh = config['w2v_zh_300d']
        self.w2v_zh_filter = config['w2v_zh_300d_filtered']

    @property
    def w2v(self):
        return self.w2v_zh['path']

    @property
    def w2v_filter(self):
        return self.w2v_zh_filter['path']


coref_config_loader = CoreferenceConfigLoader(name='best_zh')
w2v_config_loader = Word2VecConfigLoader()
