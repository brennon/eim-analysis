#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# from tensorflow.contrib.estimator import stop_if_no_decrease_hook

RANDOM_SEED = 42

tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = ' ,id,age,concentration,hearing_impairments,musical_expertise,nationality,artistic,fault,imagination,lazy,nervous,outgoing,reserved,stress,thorough,trusting,activity,engagement,familiarity,like_dislike,positivity,tension,sex,hallelujah_reaction,location,language,music_pref_none,music_pref_hiphop,music_pref_dance,music_pref_world,music_pref_rock,music_pref_pop,music_pref_classical,music_pref_jazz,music_pref_folk,music_pref_traditional_irish'.split(',')
LABEL_COLUMN = 'hallelujah_reaction'
KEY_FEATURE_COLUMN = None
DEFAULTS = [[], [''], [], [], [], [], [''], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [''], [0], [''], [''], [], [], [], [], [], [], [], [], [], []]

# These are the raw input columns, and will be provided for prediction also
INPUT_COLUMNS = [
    # Categorical features
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('language', ['N/A', 'en', 'zh_TW'])),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('location', ['dublin', 'taipei_city', 'taichung_city'])),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('nationality', ['other', 'irish', 'british', 'taiwanese', 'indonesian', 'japanese', 'chinese', 'singaporean', 'american', 'algerian', 'thai'])),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('sex', ['female', 'male'])),

    # Numeric features
    tf.feature_column.numeric_column('activity'),
    tf.feature_column.numeric_column('artistic'),
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('concentration'),
    tf.feature_column.numeric_column('engagement'),
    tf.feature_column.numeric_column('familiarity'),
    tf.feature_column.numeric_column('fault'),
    tf.feature_column.numeric_column('hearing_impairments'),
    tf.feature_column.numeric_column('imagination'),
    tf.feature_column.numeric_column('lazy'),
    tf.feature_column.numeric_column('like_dislike'),
    tf.feature_column.numeric_column('musical_expertise'),
    tf.feature_column.numeric_column('music_pref_classical'),
    tf.feature_column.numeric_column('music_pref_dance'),
    tf.feature_column.numeric_column('music_pref_folk'),
    tf.feature_column.numeric_column('music_pref_hiphop'),
    tf.feature_column.numeric_column('music_pref_jazz'),
    tf.feature_column.numeric_column('music_pref_none'),
    tf.feature_column.numeric_column('music_pref_pop'),
    tf.feature_column.numeric_column('music_pref_rock'),
    tf.feature_column.numeric_column('music_pref_traditional_irish'),
    tf.feature_column.numeric_column('music_pref_world'),
    tf.feature_column.numeric_column('nervous'),
    tf.feature_column.numeric_column('outgoing'),
    tf.feature_column.numeric_column('positivity'),
    tf.feature_column.numeric_column('reserved'),
    tf.feature_column.numeric_column('stress'),
    tf.feature_column.numeric_column('tension'),
    tf.feature_column.numeric_column('thorough'),
    tf.feature_column.numeric_column('trusting')
    
    # Engineered features that are created in the input_fn
]

# Build the estimator
def build_estimator(model_dir, nbuckets, hidden_units, learning_rate=0.001, beta1=0.9, beta2=0.999, dropout=None, activation_function='relu', checkpoint_secs=90):
    """
    Build an estimator starting from INPUT COLUMNS.
    These include feature transformations and synthetic features.
    The model is a wide-and-deep model.
    """

    # Input columns
    (language, location, nationality, sex, activity, artistic, age, concentration, engagement, familiarity, fault, hearing_impairments, imagination, lazy, like_dislike, musical_expertise, music_pref_classical, music_pref_dance, music_pref_folk, music_pref_hiphop, music_pref_jazz, music_pref_none, music_pref_pop, music_pref_rock, music_pref_traditional_irish, music_pref_world, nervous, outgoing, positivity, reserved, stress, tension, thorough, trusting) = INPUT_COLUMNS

    # Wide columns and deep columns
    wide_columns = [
        # Feature crosses

        # Sparse columns

        # Anything with a linear relationship
    ]

    deep_columns = [
        # Embedding columns

        # Numeric columns
        # language, 
        location, 
        nationality, 
        sex, 
        activity, 
        # artistic, 
        age,
        concentration,
        # engagement,
        familiarity,
        fault,
        hearing_impairments,
        imagination,
        lazy,
        like_dislike,
        musical_expertise,
        # music_pref_classical,
        music_pref_dance,
        music_pref_folk,
        music_pref_hiphop,
        music_pref_jazz,
        music_pref_none,
        # music_pref_pop,
        music_pref_rock,
        music_pref_traditional_irish,
        music_pref_world,
        nervous,
        outgoing,
        positivity,
        # reserved,
        stress,
        tension,
        thorough,
        trusting
    ]
    
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )

    run_config = tf.estimator.RunConfig(
        tf_random_seed=RANDOM_SEED,
        save_checkpoints_secs=checkpoint_secs,  # Save checkpoints every 90 seconds
        keep_checkpoint_max=100,       # Retain the 10 most recent checkpoints.
        session_config=session_config
    )

    activation_functions = {
        'elu': tf.nn.elu,
        'relu': tf.nn.relu,
        'leaky_relu': tf.nn.leaky_relu
    }

    optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
    
    estimator = tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config,
        optimizer=optimizer,
        dropout=dropout,
        activation_fn=activation_functions[activation_function]
    )

    estimator = tf.contrib.estimator.add_metrics(estimator, additional_metrics)
    return estimator

# Create input function to load data into datasets
def read_dataset(args, mode):
    batch_size = args['train_batch_size']

    if mode == tf.estimator.ModeKeys.TRAIN:
        input_paths = args['train_data_paths']
    else:
        input_paths = args['eval_data_paths']

    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return features, label

        # Create list of files that match pattern
        file_list = tf.gfile.Glob(input_paths)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # indefinitely
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1  # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    return _input_fn

# Create estimator train and evaluate function
def train_and_evaluate(args):
    hidden_units = [str(args['num_nodes']) for l in range(args['num_layers'])]

    estimator = build_estimator(args['output_dir'], args['nbuckets'], hidden_units,
                                args['learning_rate'], args['beta1'], args['beta2'], args['dropout'],
                                args['activation_function'])
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(args, tf.estimator.ModeKeys.TRAIN),
        max_steps = args['train_steps'])
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(args, tf.estimator.ModeKeys.EVAL),
        steps = args['eval_steps'],
        start_delay_secs = 5,
        throttle_secs = 5)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if args['optimize']:
        return estimator.evaluate(read_dataset(args, tf.estimator.ModeKeys.EVAL), steps=1)
    else:
        return

def additional_metrics(labels, predictions):
    precision, precision_op = tf.metrics.precision(labels, predictions['class_ids'])
    recall, recall_op = tf.metrics.recall(labels, predictions['class_ids'])
    f1 = 2. / ((1. / precision) + (1. / recall))
    return {
        'f1_score': (f1, tf.group(precision_op, recall_op)),
        'true_positives': tf.metrics.true_positives(labels, predictions['class_ids']),
        'true_negatives': tf.metrics.true_negatives(labels, predictions['class_ids']),
        'false_positives': tf.metrics.false_positives(labels, predictions['class_ids']),
        'false_negatives': tf.metrics.false_negatives(labels, predictions['class_ids'])
    }
