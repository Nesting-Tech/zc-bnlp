#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf

import bnlp.model.coreference.coref_model as cm
# from bnlp.config.bnlp_config import coref_config_loader
from bnlp.config.bnlp_config import CoreferenceConfigLoader
from bnlp.trainer import copy_checkpoint
from bnlp.utils.cal_tool import make_summary
from bnlp import project_dir


def train(name):
    config = CoreferenceConfigLoader(name=name).config

    log_dir = os.path.join(project_dir, 'data', config['log_dir'])

    report_frequency = config["report_frequency"]
    eval_frequency = config["eval_frequency"]

    model = cm.CorefModel(config)
    saver = tf.train.Saver()

    writer = tf.summary.FileWriter(log_dir, flush_secs=20)

    max_f1 = 0
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.start_enqueue_thread(session)
        accumulated_loss = 0.0

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from: {}".format(ckpt.model_checkpoint_path))
            saver.restore(session, ckpt.model_checkpoint_path)

        initial_time = time.time()
        while True:
            tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
            accumulated_loss += tf_loss

            if tf_global_step % report_frequency == 0:
                total_time = time.time() - initial_time
                steps_per_second = tf_global_step / total_time

                average_loss = accumulated_loss / report_frequency
                print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
                writer.add_summary(make_summary({"loss": average_loss}), tf_global_step)
                accumulated_loss = 0.0

            if tf_global_step % eval_frequency == 0:
                saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
                eval_summary, eval_f1 = model.evaluate(session)

                if eval_f1 > max_f1:
                    max_f1 = eval_f1
                    copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)),
                                    os.path.join(log_dir, "model.max.ckpt"))

                writer.add_summary(eval_summary, tf_global_step)
                writer.add_summary(make_summary({"max_eval_f1": max_f1}), tf_global_step)

                print("[{}] evaL_f1={:.2f}, max_f1={:.2f}".format(tf_global_step, eval_f1, max_f1))
