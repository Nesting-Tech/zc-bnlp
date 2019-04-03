# -*- coding: utf-8 -*-
"""
    Created by JoeYip on 29/03/2019

    :copyright: (c) 2019 by nesting.xyz
    :license: BSD, see LICENSE for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time

import tensorflow as tf
import bnlp.model.coreference.coref_model as cm
from bnlp.config.bnlp_config import coref_config_loader
from bnlp.config.bnlp_config import CoreferenceConfigLoader
from bnlp.trainer import copy_checkpoint, CHECKPOINT_PATTERN, TMP_CHECKPOINT_FILE_SUFFIX
from bnlp import project_dir

"""
一边训练一边评估模型
"""


def continuous_evaluate():
    config = coref_config_loader.config
    model = cm.CorefModel(config)

    saver = tf.train.Saver()
    log_dir = os.path.join(project_dir, 'data', config["log_dir"])
    writer = tf.summary.FileWriter(log_dir, flush_secs=20)
    evaluated_checkpoints = set()
    max_f1 = 0
    checkpoint_pattern = re.compile(CHECKPOINT_PATTERN)

    with tf.Session() as session:
        while True:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path and ckpt.model_checkpoint_path not in evaluated_checkpoints:
                print("Evaluating {}".format(ckpt.model_checkpoint_path))

                # Move it to a temporary location to avoid being deleted by the training supervisor.
                tmp_checkpoint_path = os.path.join(log_dir, TMP_CHECKPOINT_FILE_SUFFIX)
                copy_checkpoint(ckpt.model_checkpoint_path, tmp_checkpoint_path)

                global_step = int(checkpoint_pattern.match(ckpt.model_checkpoint_path).group(1))
                saver.restore(session, ckpt.model_checkpoint_path)

                eval_summary, f1 = model.evaluate(session)

                if f1 > max_f1:
                    max_f1 = f1
                    copy_checkpoint(tmp_checkpoint_path, os.path.join(log_dir, TMP_CHECKPOINT_FILE_SUFFIX))

                print("Current max F1: {:.2f}".format(max_f1))

                writer.add_summary(eval_summary, global_step)
                print("Evaluation written to {} at step {}".format(log_dir, global_step))

                evaluated_checkpoints.add(ckpt.model_checkpoint_path)
                sleep_time = 60
            else:
                sleep_time = 10
            print("Waiting for {} seconds before looking for next checkpoint.".format(sleep_time))
            time.sleep(sleep_time)


"""
评估模型
"""


def evaluate(name):
    config = CoreferenceConfigLoader(name=name).config
    model = cm.CorefModel(config)
    with tf.Session() as session:
        model.restore(session)
        model.evaluate(session, official_stdout=True)
