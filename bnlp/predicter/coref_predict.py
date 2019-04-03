from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf

import bnlp.model.coreference.coref_model as cm
from bnlp.config.bnlp_config import coref_config_loader


def predict(input, output):
    # # Input file in .jsonlines format.
    # input_filename = sys.argv[2]
    #
    # # Predictions will be written to this file in .jsonlines format.
    # output_filename = sys.argv[3]

    model = cm.CorefModel(coref_config_loader.config)

    with tf.Session() as session:
        model.restore(session)

        with open(output, "w") as output_file:
            with open(input) as input_file:
                for example_num, line in enumerate(input_file.readlines()):
                    example = json.loads(line)
                    tensorized_example = model.tensorize_example(example, is_training=False)
                    feed_dict = {i: t for i, t in zip(model.input_tensors, tensorized_example)}
                    _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
                        model.predictions, feed_dict=feed_dict)
                    predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
                    example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                                    predicted_antecedents)

                    output_file.write(json.dumps(example))
                    output_file.write("\n")
                    if example_num % 100 == 0:
                        print("Decoded {} examples.".format(example_num + 1))
