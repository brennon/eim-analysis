import datetime
import logging
import os.path
import subprocess

SCRIPT_NAME = 'basic_features_basic_dnn_classifier'

GCLOUD_CMD = 'C:\\Users\\bortzb\\AppData\\Local\\Google\\Cloud SDK\\google-cloud-sdk\\bin\\gcloud.cmd'
GSUTIL_CMD = 'C:\\Users\\bortzb\\AppData\\Local\\Google\\Cloud SDK\\google-cloud-sdk\\bin\\gsutil.cmd'

REGION = 'us-central1'
BUCKET = 'eim-muse'
PROJECT = 'eim-muse'
OUTPUT_DIR = 'gs://{}/analysis/hallelujah-effect/samples/{}'.format(BUCKET, SCRIPT_NAME)

TRAIN_N = 0

#####################
# Configure logging #
#####################

logger = logging.getLogger(SCRIPT_NAME)
logger.setLevel(logging.DEBUG)

# Create file handler which logs even debug messages
now = datetime.datetime.now()
now_string = '{}{:02}{:02}{:02}{:02}{:02}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
fh = logging.FileHandler('{}-{}.log'.format(SCRIPT_NAME, now_string))
fh.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

#################
# Configure GCP #
#################

def configure_gcp():

    # Set environment variables
    os.environ['PROJECT'] = PROJECT
    os.environ['BUCKET'] = BUCKET
    os.environ['REGION'] = REGION
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../../../eim-muse-78daa688d77f.json'

    # Set GCP project and compute region
    subprocess.call([GCLOUD_CMD, 'config', 'set', 'project', PROJECT])
    subprocess.call([GCLOUD_CMD, 'config', 'set', 'compute/region', REGION])

    # Clear out destination directory in bucket
    try:
        subprocess.call('{} -m rm -rf {}'.format(GSUTIL_CMD, OUTPUT_DIR))
    except:
        pass

###############
# Query setup #
###############

def create_query(phase, EVERY_N):
    """
    phase: 1 = train 2 = valid
    """
    base_query = 'SELECT * FROM `eim-muse.hallelujah_effect.full_hallelujah_trials_cleaned_imputed`'

    if EVERY_N == None:
        if phase < 2:
            # Training
            query = "{0} WHERE MOD(FARM_FINGERPRINT(id), 10) < 7".format(base_query)
        else:
            # Validation
            query = "{0} WHERE MOD(FARM_FINGERPRINT(id), 10) >= 7".format(base_query)
    else:
        query = "{0} WHERE MOD(FARM_FINGERPRINT(id), {1}) = {2}".format(base_query, EVERY_N, phase)

    return query

#########################################
# Run queries and send CSV files to GCP #
#########################################

def collect_data():

    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT)

    for n, phase in enumerate(['train', 'eval']):
        logger.info('Processing {} dataset'.format(phase))
        query = create_query(n+1, None)
        query_job = client.query(query)
        job_result = query_job.result()
        df = job_result.to_dataframe()
        train_n = df.shape[0]

        local_file = os.path.join('.', 'sample', '{}.csv'.format(phase))
        with open(local_file, 'w') as f:
            df.to_csv(f, header=False)

        gcp_file = OUTPUT_DIR + '/{}.csv'.format(phase)
        subprocess.call([GSUTIL_CMD, 'cp', local_file, gcp_file])
        logger.info('CSV file for {} dataset written to {}'.format(phase, gcp_file))

    #########################################
    # Save all data so that we have headers #
    #########################################

    query = create_query(0, 1)
    query_job = client.query(query)
    job_result = query_job.result()
    df = job_result.to_dataframe()

    local_file = os.path.join('.', 'sample', 'all_with_headers.csv')
    with open(local_file, 'w') as f:
        df.to_csv(f, header=True)

    gcp_file = OUTPUT_DIR + '/all_with_headers.csv'
    subprocess.call([GSUTIL_CMD, 'cp', local_file, gcp_file])
    logger.info('CSV file for full dataset (with headers) written to {}'.format(phase, gcp_file))

    return train_n

#########################
# Test training locally #
#########################

def train_locally():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    subprocess.call([
        'python',
        '-m',
        'trainer.trainer.task',
        '--train_data_paths=%cd%\\sample\\train*',
        '--eval_data_paths=%cd%\\sample\\eval*',
        '--output_dir=%cd%\\model_trained',
        '--train_steps=5',
        '--job-dir=C:\\Windows\\Temp'
    ])

if __name__ == '__main__':
    configure_gcp()
    TRAIN_N = collect_data()
    # train_locally()