
# Higher-order Coreference Resolution with Coarse-to-fine Inference

## Introduction
This repository contains the code for replicating results from

* [Higher-order Coreference Resolution with Coarse-to-fine Inference](https://arxiv.org/abs/1804.05392)
* [Kenton Lee](http://kentonl.com/), [Luheng He](https://homes.cs.washington.edu/~luheng), and [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz)
* In NAACL 2018

## Getting Started

* Install python (either 2 or 3) requirements: `pip install -r requirements.txt`
* Build custom kernels by running `python main.py build`.
  * There are 3 platform-dependent ways to build custom TensorFlow kernels. Please comment/uncomment the appropriate lines in the script.
* Download pretrained word embeddings and model in the following link:
  * Link: https://pan.baidu.com/s/1PCE-Et7qpBbjL6wT1apCwQ
  * Password: dbze
  * Put `model.max.ckpt.index` and `model.max.ckpt.data-00000-of-00001` at `data/logs/demo_zh/` directory. The others put at `data/` directory.

## Demo Instructions

* Command-line demo: `python demo_zh.py demo_zh`
* To run the demo with other experiments, replace `demo_zh` with your configuration name.


## Training Instructions
* To train your own model, run `bash setup_conll2012.sh` that is locate at `bin/` to create conll-2012 data set.
  * This assumes access to OntoNotes 5.0. Please edit the `ontonotes_path` variable.
  * This script need to be run in python2 environment, using `virtulenv` is a good choice.
* Reconstruct conll-2012 data set with `python main.py prepare --lang chinese`
* Experiment configurations are found in `coreference.conf`
* Choose an experiment that you would like to run, e.g. `best_zh`
* Training: `python main.py train --model coref --name best_zh`
* Results are stored in the `data/logs/best_zh` directory and can be viewed via TensorBoard.
* Evaluation: `python main.py evaluate --model coref --name best_zh `


## Batched Prediction Instructions

* Create a file where each line is in the following json format (make sure to strip the newlines so each line is well-formed json):
```
{
  "clusters": [],
  "doc_key": "nw",
  "sentences": [["This", "is", "the", "first", "sentence", "."], ["This", "is", "the", "second", "."]],
  "speakers": [["spk1", "spk1", "spk1", "spk1", "spk1", "spk1"], ["spk2", "spk2", "spk2", "spk2", "spk2"]]
}
```
  * `clusters` should be left empty and is only used for evaluation purposes.
  * `doc_key` indicates the genre, which can be one of the following: `"bc", "bn", "mz", "nw", "pt", "tc", "wb"`
  * `speakers` indicates the speaker of each word. These can be all empty strings if there is only one known speaker.
* Run `python coref_predict.py <experiment> <input_file> <output_file>`, which outputs the input jsonlines with predicted clusters.

## Other Quirks

* It does not use GPUs by default. Instead, it looks for the `GPU` environment variable, which the code treats as shorthand for `CUDA_VISIBLE_DEVICES`.
* The training runs indefinitely and needs to be terminated manually. The model generally converges at about 400k steps.

## Acknowledgement
* This repository is copy from: [e2e-coref](https://github.com/kentonl/e2e-coref). Thanks a lot~