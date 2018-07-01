#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from tensorflow_transform.saved import input_fn_maker, saved_transform_io
from tensorflow_transform.tf_metadata import metadata_io

tf.logging.set_verbosity(tf.logging.INFO)

# CSV_COLUMNS = 'age,activity,hallelujah_reaction'.split(',')
LABEL_COLUMN = 'hallelujah_reaction'
KEY_FEATURE_COLUMN = None
# DEFAULTS = [[0.0], ['Sun'], [0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

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
def build_estimator(model_dir, nbuckets, hidden_units):
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
        language, location, nationality, sex, activity, artistic, age, concentration, engagement, familiarity, fault, hearing_impairments, imagination, lazy, like_dislike, musical_expertise, music_pref_classical, music_pref_dance, music_pref_folk, music_pref_hiphop, music_pref_jazz, music_pref_none, music_pref_pop, music_pref_rock, music_pref_traditional_irish, music_pref_world, nervous, outgoing, positivity, reserved, stress, tension, thorough, trusting
    ]
    
    checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs = 5,  # Save checkpoints every 5 seconds
        keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
    )
    
    estimator = tf.estimator.DNNLinearCombinedClassifier(
        model_dir = model_dir,
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_hidden_units = hidden_units,
        config=checkpointing_config)
    
    # estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)
    return estimator

# Create serving input function to be able to serve predictions
def make_serving_input_fn(args):
  transform_savedmodel_dir = (
        os.path.join(args['metadata_path'], 'transform_fn'))

  def _input_fn():
    # Placeholders for all the raw inputs; a lot of inputs are missing here
    feature_placeholders = {
      column_name: tf.placeholder(tf.float32, [None]) for column_name in 'age,activity'.split(',')
    }

    # transform using the saved model in transform_fn
    _, features = saved_transform_io.partially_apply_saved_transform(
      transform_savedmodel_dir,
      feature_placeholders
    )
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

  return _input_fn

# Create input function to load data into datasets
def read_dataset(args, mode):
    batch_size = args['train_batch_size']
    if mode == tf.estimator.ModeKeys.TRAIN:
        input_paths = args['train_data_paths']
    else:
        input_paths = args['eval_data_paths']
    
    transformed_metadata = metadata_io.read_metadata(
        os.path.join(args['metadata_path'], 'transformed_metadata'))

    return input_fn_maker.build_training_input_fn(
        metadata = transformed_metadata,
        file_pattern = (
          input_paths[0] if len(input_paths) == 1 else input_paths),
        training_batch_size = batch_size,
        label_keys = [LABEL_COLUMN],
        reader = gzip_reader_fn,
        key_feature_name = KEY_FEATURE_COLUMN,
        randomize_input = (mode != tf.estimator.ModeKeys.EVAL),
        num_epochs = (1 if mode == tf.estimator.ModeKeys.EVAL else None)) 

# Create estimator train and evaluate function
def train_and_evaluate(args):
    estimator = build_estimator(args['output_dir'], args['nbuckets'], args['hidden_units'].split(' '))
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(args, tf.estimator.ModeKeys.TRAIN),
        max_steps = args['train_steps'])
    exporter = tf.estimator.LatestExporter(
        'exporter', make_serving_input_fn(args))
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(args, tf.estimator.ModeKeys.EVAL),
        steps = args['eval_steps'],
        exporters = exporter,
        start_delay_secs = 5,
        throttle_secs = 5)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# If we want to use TFRecords instead of CSV
def gzip_reader_fn():
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
            compression_type = tf.python_io.TFRecordCompressionType.GZIP))

def add_eval_metrics(labels, predictions):
    # pred_values = predictions['predictions']
    return {
        'confusion_matrix': tf.confusion_matrix(labels, predictions)
    }
