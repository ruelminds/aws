import pyarrow
import pyarrow.parquet as pq
import signal
import tarfile
import sys
import boto3
import io
import json
import re
import pandas as pd
import uuid
from collections import namedtuple
from functools import partial
from itertools import zip_longest
import os
import timeit
import time
import traceback
import datetime
from dateutil import rrule
from glob import glob

def get_batches(iterable, batch_size, fillvalue=None):
    args = [iter(iterable)] * batch_size
    return zip_longest(*args, fillvalue=fillvalue)

s3_client = None

def get_s3_client():
    global s3_client
    if (s3_client is None):
        s3_client = boto3.client('s3')
    return s3_client

def get_matching_s3_objects(bucket, prefix="", suffix="", max_objects=20000000):
    s3 = get_s3_client()

    """
    Generate objects in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch objects whose key starts with
        this prefix (optional).
    :param suffix: Only fetch objects whose keys end with
        this suffix (optional).
    """
    try:
        paginator = s3.get_paginator("list_objects_v2")

        kwargs = {'Bucket': bucket}

        # We can pass the prefix directly to the S3 API.  If the user has passed
        # a tuple or list of prefixes, we go through them one by one.
        if isinstance(prefix, str):
            prefixes = (prefix, )
        else:
            prefixes = prefix

        num_objects_retrieved = 0
        for key_prefix in prefixes:
            kwargs["Prefix"] = key_prefix

            for page in paginator.paginate(**kwargs):
                try:
                    contents = page["Contents"]
                except KeyError:
                    return

                for obj in contents:
                    key = obj["Key"]

                    if key.endswith(suffix) and obj['Size'] < 100000000:
                        num_objects_retrieved = num_objects_retrieved + 1
                        yield obj

                    if (num_objects_retrieved > max_objects):
                        return

    except Exception as e:
        print(e)
        print(f"Error fetching matching S3 objects for bucket {bucket}. prefix {prefix}. suffix {suffix}")

def write_pandas_parquet_to_s3(df, bucketName, keyName, fileName):
    required_columns = ['index', 'timestamp', 'speed', 'rpm', 'throttle',
       'distanceLogInterval', 'litresLogInterval', 'co2LogInterval',
       'xAxisAcceleration', 'yAxisAcceleration', 'zAxisGyroscope',
       'massAirFlow', 'massAirPressure', 'calculatedEngineLoad',
       'efficiencyMetric', 'batteryVoltageMillivolts', 'speedlimit', 'tripId',
       'createdAt', 'updatedAt', 'latitude', 'longitude']

    for required_column in required_columns:
        if (required_column not in df.columns):
            print(f"missing {required_column}")
            df[required_column] = None

    schema = pyarrow.schema([
        ('createdAt', pyarrow.timestamp('ms')),
        ('updatedAt', pyarrow.timestamp('ms')),
        ('timestamp', pyarrow.timestamp('ms')),
        ('index', pyarrow.int32()),
        ('speed', pyarrow.int32()),
        ('throttle', pyarrow.int32()),
        ('massAirFlow', pyarrow.int32()),
        ('massAirPressure', pyarrow.int32()),
        ('calculatedEngineLoad', pyarrow.int32()),
        ('efficiencyMetric', pyarrow.int32()),
        ('speedlimit', pyarrow.int32()),
        ('rpm', pyarrow.float64()),
        ('distanceLogInterval', pyarrow.float64()),
        ('litresLogInterval', pyarrow.float64()),
        ('co2LogInterval', pyarrow.float64()),
        ('xAxisAcceleration', pyarrow.float64()),
        ('yAxisAcceleration', pyarrow.float64()),
        ('zAxisGyroscope', pyarrow.float64()),
        ('batteryVoltageMillivolts', pyarrow.float64()),
        ('latitude', pyarrow.float64()),
        ('longitude', pyarrow.float64()),
        ('tripId', pyarrow.string())
    ])
    table = pyarrow.Table.from_pandas(df, schema=schema)
    pq.write_table(table, fileName)
    file_size = os.path.getsize(fileName)
    print(f"Combined parquet file size: {file_size / 1000}kb")

    # upload to s3
    s3 = get_s3_client()
    BucketName = bucketName
    with open(fileName, 'rb') as f:
       object_data = f.read()
       s3.put_object(Body=object_data, Bucket=BucketName, Key=keyName)
    print("uploaded to s3")

processed_s3_keys = list()

# Read single parquet file from S3
def pd_read_s3_parquet(key, bucket):
    s3_client = get_s3_client()

    local_file = '/tmp/local.parquet'
    s3_client.download_file(bucket, key, local_file)
    return pd.read_parquet(local_file)

def process_s3_object(s3_object, bucket_name):
    try:
        df = pd_read_s3_parquet(s3_object['Key'], bucket=bucket_name)
        global processed_s3_keys
        processed_s3_keys.append(s3_object['Key'])
        return df
    except Exception as err:
        traceback.print_tb(err.__traceback__)
        return pd.DataFrame()

def get_s3_objects_to_process(folder_path, bucket_name, num_objects=5000):
    s3_objects = get_matching_s3_objects(bucket_name, folder_path, '', num_objects)
    fetched_s3_objects = list(s3_objects)

    return fetched_s3_objects

# Read multiple parquets from a folder on S3 generated by data ingest lambda
def pd_read_s3_multiple_parquets(s3_objects, bucket_name, folder_path):
    if len(s3_objects) == 0:
        return None

    dfs = [process_s3_object(s3_object, bucket_name) for s3_object in s3_objects]

    return pd.concat(dfs, ignore_index=True)

def get_folder_for_processed_partition(bucket_path):
    # e.g. Processed/TripDetails/2020-06-02T16/6d7499c1-2c1e-47f7-89d9-708852fa914a.parquet
    bucket_path_without_processed = bucket_path.replace('Processed/', '')

    if ('year=' in bucket_path_without_processed):
        # e.g. Processed/TripDetails/year=2020/month=04/day=27/hour=17/blah.parquet
        # drop the filename and keep the existing folder path
        return re.sub('(.*)/.*.parquet', r'\1', bucket_path_without_processed)

    date_time = re.sub('(.*)/(20.*)/([^/]*)/?.*.parquet', r'\2', bucket_path_without_processed)
    # e.g. TripDetails would be the entity
    entity = re.sub('(.*)/(20.*)/([^/]*)/?.*.parquet', r'\1', bucket_path_without_processed)

    date_time_parts = date_time.split('T')
    date_parts = date_time_parts[0].split('-')
    year = date_parts[0]
    month = date_parts[1]
    day = date_parts[2]
    hour = date_time_parts[1] if len(date_time_parts) > 1 else 1

    # Goal TripDetails/year=2020/month=04/day=27/hour=17
    return f"{entity}/year={year}/month={month}/day={day}/hour={hour}"

efs_path = "/mnt/efs"
in_progress_path = f"{efs_path}/in_progress"

def mark_s3_path_as_in_progress(path):
    if not os.path.exists(f"{in_progress_path}/{path}"):
        os.makedirs(f"{in_progress_path}/{path}")

def is_s3_path_marked_in_progress(path):
    return os.path.isdir(f"{in_progress_path}/{path}")

def mark_s3_path_not_in_progress(path):
    if os.path.exists(f"{in_progress_path}/{path}"):
        os.rmdir(f"{in_progress_path}/{path}")

current_path_being_processed = ''

def can_process_s3_path(s3_path):
    if (is_s3_path_marked_in_progress(s3_path)):
        last_modified_time_of_marker = datetime.datetime.fromtimestamp(max(os.stat(root).st_mtime for root,_,_ in os.walk(f"{in_progress_path}/{s3_path}")))

        timesince = datetime.datetime.now() - last_modified_time_of_marker
        minutes_since_s3_path = int(timesince.total_seconds() / 60)

        MAX_IN_PROGRESS_MINUTES = 15
        return minutes_since_s3_path > MAX_IN_PROGRESS_MINUTES
    return True

def get_non_processed_folders(bucket_name):
    folder_paths = list()
    s3_objects_in_non_processed_dir = list(get_matching_s3_objects(bucket_name, prefix='TripDetails', suffix="", max_objects=1000))
    for obj in s3_objects_in_non_processed_dir:
        folder_path = re.sub('/[^/]*.parquet', '', obj['Key'])

        if (folder_path not in folder_paths):
            folder_paths.append(folder_path)

    return folder_paths

def get_processed_folders(bucket_name):
    folder_paths = list()
    s3_objects_in_processed_dir = list(get_matching_s3_objects(bucket_name, prefix='Processed', suffix="", max_objects=1000))
    for obj in s3_objects_in_processed_dir:
        folder_path = f"Processed/{get_folder_for_processed_partition(obj['Key'])}"
        if (folder_path not in folder_paths):
            folder_paths.append(folder_path)

    return folder_paths

def get_potential_s3_paths(bucket_name):
    start_date = datetime.datetime.now() - datetime.timedelta(days=1)
    end_date = start_date + datetime.timedelta(days=2)
    valid_paths = list()

    non_processed_paths = get_non_processed_folders(bucket_name)
    for path in non_processed_paths:
        valid_paths.append(path)

    processed_paths = get_processed_folders(bucket_name)
    for path in processed_paths:
        valid_paths.append(path)

    if (len(non_processed_paths) > 0):
        for time_checkpoint in rrule.rrule(rrule.HOURLY, dtstart=start_date, until=end_date):
            if (len(valid_paths) > 15):
                break

            path = f"TripDetails/{time_checkpoint.strftime('%Y-%m-%dT%H')}"

            if (not can_process_s3_path(path)):
                continue

            s3_objects_for_path = list(get_matching_s3_objects(bucket_name, prefix=path, suffix="", max_objects=1))
            if (len(s3_objects_for_path) > 0):
                valid_paths.append(path)

    print(f"valid_paths {valid_paths}")
    result = list(set(valid_paths))
    return result

def is_about_to_timeout(context):
    global BUFFER_TIME_SECONDS_TO_CLEAN_UP
    time_out_remaining_seconds = int(context.get_remaining_time_in_millis() / 1000) - BUFFER_TIME_SECONDS_TO_CLEAN_UP
    return time_out_remaining_seconds <= 0

def combine_small_parquet_files(s3_object_path_prefix, BUCKET_NAME, context):
    if (is_about_to_timeout(context)):
        print("uh oh. About to time out")
        return
    if not os.path.exists(in_progress_path):
        os.makedirs(in_progress_path)

    dirs_in_progress = glob(f"{in_progress_path}/*/")
    print("Paths being processed")
    print(json.dumps(dirs_in_progress))

    potential_s3_paths = get_potential_s3_paths(BUCKET_NAME)
    print(f"potential_s3_paths {potential_s3_paths}")

    empty_folders = []
    global processed_s3_keys
    for folder_path in potential_s3_paths:
        completed = False
        while(not completed):
            if (is_about_to_timeout(context)):
                print("uh oh. About to time out")
                return

            if (folder_path in empty_folders or not can_process_s3_path(folder_path)):
                completed = True
                print(f"empty folder {folder_path in empty_folders} can_process_s3_path(folder_path) {can_process_s3_path(folder_path)}" )
                continue
            try:
                tic = timeit.default_timer()
                mark_s3_path_as_in_progress(folder_path)
                global current_path_being_processed
                current_path_being_processed = folder_path
                print(folder_path)

                num_objects = 200 if "Processed" in folder_path else 6000
                s3_objects_to_process = get_s3_objects_to_process(folder_path, BUCKET_NAME, num_objects)

                print(f"files {len(s3_objects_to_process)}")

                pandas_dataframe = pd_read_s3_multiple_parquets(s3_objects_to_process, BUCKET_NAME, folder_path)
                if (pandas_dataframe is None):
                    print("pd df is empty")
                    empty_folders.append(folder_path)
                    completed = True
                    continue

                bucket_path = s3_objects_to_process[0]['Key']
                path_with_partition_keys = get_folder_for_processed_partition(bucket_path)
                if ("Processed" in folder_path):
                    merged_file_s3_key = f"Reprocessed/{path_with_partition_keys}/{uuid.uuid4()}.parquet"
                else:
                    merged_file_s3_key = f"Processed/{path_with_partition_keys}/{uuid.uuid4()}.parquet"

                write_pandas_parquet_to_s3(
                    pandas_dataframe,
                    BUCKET_NAME,
                    merged_file_s3_key,
                    "/tmp/file.parquet"
                )
                object_batches = get_batches(s3_objects_to_process, 1000)

                s3_client = get_s3_client()

                for object_batch in object_batches:
                    print(f"delete_objects {object_batch}")
                    objects = [{ "Key": s3_obj["Key"] } for s3_obj in object_batch if s3_obj]
                    s3_client.delete_objects(
                        Bucket=BUCKET_NAME,
                        Delete={
                            'Objects': objects,
                            'Quiet': False
                        }
                    )

                print("deleted")
                mark_s3_path_not_in_progress(folder_path)
                toc = timeit.default_timer()
                time_to_process = toc - tic
                print(f"Time to process {len(s3_objects_to_process)} parquet files in {folder_path}: {time_to_process} seconds")
            except Exception as e:
                exception_message = str(e)
                traceback.print_tb(e.__traceback__)
                print(f"handled exception {exception_message}")
                if (exception_message == 'Time exceeded'):
                    mark_s3_path_not_in_progress(current_path_being_processed)
                    return

                print(f"Error with {folder_path} {e}")
                raise

def timeout_handler(_signal, _frame):
    '''Handle SIGALRM'''
    raise Exception('Time exceeded')

BUFFER_TIME_SECONDS_TO_CLEAN_UP = 15

def handler(event, context):
    time_out_remaining_seconds = int(context.get_remaining_time_in_millis() / 1000) - BUFFER_TIME_SECONDS_TO_CLEAN_UP
    signal.alarm(time_out_remaining_seconds)
    signal.signal(signal.SIGALRM, timeout_handler)
    BUCKET_NAME = os.environ['DATA_DUMP_BUCKET_NAME']
    try:
        combine_small_parquet_files('TripDetails', BUCKET_NAME, context)
    except Exception as e:
        global current_path_being_processed
        exception_message = str(e)
        if (exception_message == 'Time exceeded'):
            mark_s3_path_not_in_progress(current_path_being_processed)
            return
