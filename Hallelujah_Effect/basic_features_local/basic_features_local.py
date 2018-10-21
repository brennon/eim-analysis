import datetime
import logging
import os.path
import subprocess
import tensorflow as tf

script_name = 'basic_features_local'
now = datetime.datetime.now()
now_string = '{}{:02}{:02}{:02}{:02}{:02}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)

#####################
# Configure logging #
#####################

logger = logging.getLogger(script_name)
logger.setLevel(logging.DEBUG)
# Create file handler which logs even debug messages
fh = logging.FileHandler('{}-{}.log'.format(script_name, now_string))
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

logger.info('Using TensorFlow version {}'.format(tf.__version__))

#################
# Configure GCP #
#################

gcloud_cmd = 'C:\\Users\\bortzb\\AppData\\Local\\Google\\Cloud SDK\\google-cloud-sdk\\bin\\gcloud.cmd'
gsutil_cmd = 'C:\\Users\\bortzb\\AppData\\Local\\Google\\Cloud SDK\\google-cloud-sdk\\bin\\gsutil.cmd'

REGION = 'us-central1'
BUCKET = 'eim-muse'
PROJECT = 'eim-muse'
OUTPUT_DIR = 'gs://{}/analysis/hallelujah-effect/samples/{}'.format(BUCKET, script_name)

os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../../eim-muse-78daa688d77f.json'

subprocess.call([gcloud_cmd, 'config', 'set', 'project', PROJECT])
subprocess.call([gcloud_cmd, 'config', 'set', 'compute/region', REGION])

###############
# Query setup #
###############

def create_query(phase, EVERY_N):
    """
    phase: 1 = train 2 = valid
    """
    base_query = """
SELECT *
FROM
  `eim-muse.hallelujah_effect.full_hallelujah_trials_cleaned_imputed`"""

    if EVERY_N == None:
        if phase < 2:
            # Training
            query = "{0} WHERE MOD(ABS(FARM_FINGERPRINT(id)), 10) < 7".format(base_query)
        else:
            # Validation
            query = "{0} WHERE MOD(ABS(FARM_FINGERPRINT(id)), 10) >= 7".format(base_query)
    else:
        query = "{0} WHERE MOD(ABS(FARM_FINGERPRINT(id)), {1}) == {2}".format(base_query, EVERY_N, phase)

    return query

# Clear out destination directory in bucket
output = None
try:
    output = subprocess.check_output('{} -m rm -rf {}'.format(gsutil_cmd, OUTPUT_DIR))
except:
    pass

#########################################
# Run queries and send CSV files to GCP #
#########################################
from google.cloud import bigquery
client = bigquery.Client(project=PROJECT)

for n, phase in enumerate(['train', 'eval']):
    query = create_query(n+1, None)
    query_job = client.query(query)
    job_result = query_job.result()
    df = job_result.to_dataframe()

    local_file = os.path.join('.', 'train.csv')
    with open(local_file, 'w') as f:
        df.to_csv(f)

    gcp_file = OUTPUT_DIR + '/{}.csv'.format(phase)
    subprocess.call([gsutil_cmd, 'mv', local_file, gcp_file])
    logger.info('CSV file for {} dataset written to {}'.format(phase, gcp_file))