#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import click

import bnlp
from bnlp.utils.text_tool import mkdirs


@click.group()
@click.pass_context
def main(context):
    pass


@main.command(help="Build custom kernels")
@click.pass_context
def build(context):
    sh = 'bash ' + bnlp.project_dir + '/bin/compile_coref.sh'
    os.system(sh)
    mkdirs(os.path.join(bnlp.project_dir, 'data', 'logs', 'demo_zh'))


@main.command(help="Evaluate data set")
@click.option("--model", default="coref", help="Support coref/NRE")
@click.option("--name", default="best_zh", help="Model config name in coreference.conf")
@click.pass_context
def evaluate(context, model, name):
    import importlib

    method = 'bnlp.trainer.{0}_evaluate'.format(model)
    print(method)
    module = importlib.import_module(method)
    module.evaluate(name)


@main.command(help="Prepare conll 2012 dataset")
@click.option("--lang", default="chinese", help="Support chinese/english/arabic")
@click.pass_context
def prepare(context, lang):
    data_dir = os.path.join(bnlp.project_dir, 'data')
    from bnlp.data.conll2012 import prepare, cache_zh
    prepare(lang, data_dir)
    cache_zh(data_dir)


@main.command(help="Train data set")
@click.option("--model", default="coref", help="Support coref/NRE")
@click.option("--name", default="best_zh", help="Model config name in coreference.conf")
@click.pass_context
def train(context, model, name):
    import importlib

    method = 'bnlp.trainer.{0}_trainer'.format(model)
    print(method)
    module = importlib.import_module(method)
    module.train(name)


@main.command(help="Test data set")
@click.option("--model", default="coref", help="Support coref/NRE")
@click.option("--continuous", default="n", help="")
@click.pass_context
def test(context, model, continuous):
    import importlib

    method = 'bnlp.trainer.{0}_evaluate.train.{1}'.format(model,
                                                          'continuous_evaluate' if continuous == 'y' else 'evaluate')
    module = importlib.import_module(method)
    module.func()


if __name__ == "__main__":
    main()
