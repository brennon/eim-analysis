# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example implementation of code to run on the Cloud ML service.
"""

import traceback
import argparse
import json
import os

from .model import train_and_evaluate

import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train_data_paths',
        help = 'GCS or local path to training data',
        required = True
    )
    parser.add_argument(
        '--train_batch_size',
        help = 'Batch size for training steps',
        type = int,
        default = 512
    )
    parser.add_argument(
        '--eval_batch_size',
        help = 'Batch size for evaluation steps',
        type = int,
        default = 512
    )
    parser.add_argument(
        '--train_steps',
        help = 'Steps to run the training job for',
        type = int,
        default = 5000
    )
    parser.add_argument(
        '--eval_steps',
        help = 'Number of steps to run evalution for at each checkpoint',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--eval_data_paths',
        help = 'GCS or local path to evaluation data',
        required = True
    )
    # TensorFlow Transform args
    parser.add_argument(
        '--metadata_path',
        help = 'GCS or local path to transformed metadata if using TFT',
        default = './preproc_tft/metadata'
    )
    # Training arguments
    parser.add_argument(
        '--nbuckets',
        help = 'Number of buckets into which to discretize lats and lons',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--hidden_units',
        help = 'Hidden layer sizes to use for DNN feature columns -- provide space-separated layers',
        type = str,
        default = "128 32 4"
    )
    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'junk'
    )
    # Eval arguments
    parser.add_argument(
        '--eval_delay_secs',
        help = 'How long to wait before running first evaluation',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--checkpoint_secs',
        help='How often (in seconds) to checkpoint',
        default=90,
        type=int
    )
    parser.add_argument(
        '--min_eval_frequency',
        help = 'Minimum number of training steps between evaluations',
        default = 1,
        type = int
    )
    parser.add_argument(
        '--learning_rate',
        help = 'Learning rate',
        default = 0.001,
        type = float
    )
    parser.add_argument(
        '--beta1',
        help = 'Adam optimizer beta_1 parameter',
        default = 0.9,
        type = float
    )
    parser.add_argument(
        '--beta2',
        help = 'Adam optimizer beta_1 parameter',
        default = 0.999,
        type = float
    )
    parser.add_argument(
        '--dropout',
        help = 'Probability a given node will be dropped',
        default = 0.,
        type = float
    )
    parser.add_argument(
        '--activation_function',
        help = 'Choice of activation function (supported choices include \'elu\', \'relu\', and \'leaky_relu\'',
        default = 'relu',
        type = str
    )
    parser.add_argument(
        '--optimize',
        help='Whether or not hyperparameter optimization should be performed',
        default=False,
        type=bool
    )
    parser.add_argument(
        '--optimize_trials',
        help='Number of hyperparameter optimization trials to be performed',
        default=5,
        type=int
    )

    args = parser.parse_args()
    arguments = args.__dict__

    # Unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    output_dir = arguments['output_dir']

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    if arguments['optimize']:
        output_dir = os.path.join(
            output_dir,
            json.loads(
                os.environ.get('TF_CONFIG', '{}')
            ).get('task', {}).get('trial', '')
        )

    arguments['output_dir'] = output_dir

    # Run the training job:
    try:
        if arguments['optimize']:
            from .optimize import optimize
            optimize(arguments)
        else:
            train_and_evaluate(arguments)
    except:
        traceback.print_exc()
