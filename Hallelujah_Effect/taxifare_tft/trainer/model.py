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

CSV_COLUMNS = 'age,activity,hallelujah_reaction'.split(',')
LABEL_COLUMN = 'hallelujah_reaction'
KEY_FEATURE_COLUMN = None
# DEFAULTS = [[0.0], ['Sun'], [0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

# These are the raw input columns, and will be provided for prediction also
INPUT_COLUMNS = [
    # Define features
    # tf.feature_column.categorical_column_with_identity('dayofweek', num_buckets = 100 ),  # some large number
    # tf.feature_column.categorical_column_with_identity('hourofday', num_buckets = 24),
    # tf.feature_column.categorical_column_with_identity('hallelujah_reaction'),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('location', ['dublin', 'taipei_city', 'taichung_city'])),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('language', ['N/A', 'en', 'zh_TW'])),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('sex', ['female', 'male'])),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('nationality', ['other', 'irish', 'british', 'taiwanese', 'indonesian', 'japanese', 'chinese', 'singaporean', 'american', 'algerian', 'thai'])),

    # Numeric columns
    # tf.feature_column.numeric_column('pickuplon'),
    # tf.feature_column.numeric_column('pickuplat'),
    # tf.feature_column.numeric_column('dropofflat'),
    # tf.feature_column.numeric_column('dropofflon'),
    # tf.feature_column.numeric_column('passengers'),
    tf.feature_column.numeric_column('activity'),
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('hearing_impairments'),
    tf.feature_column.numeric_column('engagement'),
    tf.feature_column.numeric_column('familiarity'),
    tf.feature_column.numeric_column('like_dislike'),
    tf.feature_column.numeric_column('positivity'),
    tf.feature_column.numeric_column('tension')
    
    # Engineered features that are created in the input_fn
    # tf.feature_column.numeric_column('latdiff'),
    # tf.feature_column.numeric_column('londiff'),
    # tf.feature_column.numeric_column('euclidean')
]

# Build the estimator
def build_estimator(model_dir, nbuckets, hidden_units):
    """
     Build an estimator starting from INPUT COLUMNS.
     These include feature transformations and synthetic features.
     The model is a wide-and-deep model.
  """

    # Input columns
    # (dayofweek, hourofday, latdiff, londiff, euclidean, plon, plat, dlon, dlat, pcount) = INPUT_COLUMNS
    (location, language, sex, activity, age, hearing_impairments, nationality, engagement, familiarity, like_dislike, positivity, tension) = INPUT_COLUMNS

    # Bucketize the lats & lons
    # latbuckets = np.linspace(0, 1.0, nbuckets).tolist()
    # lonbuckets = np.linspace(0, 1.0, nbuckets).tolist()
    # b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    # b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    # b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    # b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)

    # Feature cross
    # ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets * nbuckets)
    # dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    # pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4 )
    # day_hr =  tf.feature_column.crossed_column([dayofweek, hourofday], 24 * 7)

    # Wide columns and deep columns.
    wide_columns = [
        # Feature crosses
        # dloc, ploc, pd_pair,
        # day_hr,

        # Sparse columns
        # dayofweek, hourofday,

        # Anything with a linear relationship
        # pcount 
    ]

    deep_columns = [
        # Embedding_column to "group" together ...
        # tf.feature_column.embedding_column(pd_pair, 10),
        # tf.feature_column.embedding_column(day_hr, 10),

        # Numeric columns
        # plat, plon, dlat, dlon,
        # latdiff, londiff, euclidean
        location, language, sex, activity, age, hearing_impairments, nationality, engagement, familiarity, like_dislike, positivity, tension
    ]
    
    checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs = 5,  # Save checkpoints every 5 seconds
        keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
    )
    
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir = model_dir,
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_hidden_units = hidden_units,
        config=checkpointing_config)

# Create serving input function to be able to serve predictions
def make_serving_input_fn_for_base64_json(args):
    raw_metadata = metadata_io.read_metadata(
        os.path.join(args['metadata_path'], 'rawdata_metadata'))
    transform_savedmodel_dir = (
        os.path.join(args['metadata_path'], 'transform_fn'))
    return input_fn_maker.build_parsing_transforming_serving_input_receiver_fn(
      raw_metadata,
      transform_savedmodel_dir,
      exclude_raw_keys = [LABEL_COLUMN])

def make_serving_input_fn(args):
  transform_savedmodel_dir = (
        os.path.join(args['metadata_path'], 'transform_fn'))

  def _input_fn():
    # placeholders for all the raw inputs
    feature_placeholders = {
      # column_name: tf.placeholder(tf.float32, [None]) for column_name in 'pickuplon,pickuplat,dropofflat,dropofflon'.split(',')
      column_name: tf.placeholder(tf.float32, [None]) for column_name in 'age,activity'.split(',')
    }
    # feature_placeholders['passengers'] = tf.placeholder(tf.int64, [None])
    # feature_placeholders['dayofweek'] = tf.placeholder(tf.string, [None])
    # feature_placeholders['hourofday'] = tf.placeholder(tf.int64, [None])
    # feature_placeholders['key'] = tf.placeholder(tf.string, [None])

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

def get_eval_metrics():
    return {
        'rmse': tflearn.MetricSpec(metric_fn=metrics.streaming_root_mean_squared_error),
        'training/hptuning/metric': tflearn.MetricSpec(metric_fn=metrics.streaming_root_mean_squared_error),
    }
