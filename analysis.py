#!/usr/bin/python3

import boto3
import collections
import datetime
import io
import gzip
import time

MSG_PREFIX = '(xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx) Method request body before transformations: '
BUCKET = 'www.neal.news'

def get_logs():
    client = boto3.client('logs')
    logGroupName = 'API-Gateway-Execution-Logs_jt5a7bev0m/beta'
    endTime = datetime.datetime.now()
    startTime = endTime - datetime.timedelta(30)

    start = response = client.start_query(
            logGroupName=logGroupName,
            startTime=int(startTime.timestamp()),
            endTime=int(endTime.timestamp()),
            queryString='fields @timestamp, @message | filter @message like /Method request body before transformations: http/ ',
            limit=10000
        )

    print(start)

    status = 'Init'
    while status not in ('Complete', 'Failed','Cancelled', 'Timeout'):
        time.sleep(15)
        results = client.get_query_results(queryId=start['queryId'])
        status = results['status']


    first_ts = '2999'
    clicks = set()
    for i, record in enumerate(results['results']):
        x = {field['field'] : field['value'] for field in record}
        first_ts = min(first_ts, x['@timestamp'])
        clicks.add(x['@message'][len(MSG_PREFIX):])
    
    first_ts = datetime.datetime.strptime(first_ts, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc)
    return clicks, first_ts


def get_docs_keys(client, oldest):

    listing = client.list_objects_v2(Bucket=BUCKET)

    for i in listing['Contents']:
        if i['LastModified'] > oldest:
            yield i['Key']


def fetch_s3(client, id):
    print("fetch_s3")
    obj = client.get_object(Bucket=BUCKET, Key=id)
    f = io.StringIO(gzip.open(obj['Body']).read().decode('utf-8'))
    return f

def main(args):
    Y, first_ts =  get_logs()
#    print(Y)
#    print(first_ts)
    s3_client = boto3.client('s3')
    X = get_docs_keys(s3_client, first_ts + datetime.timedelta(-1))

    for k in X:
        print(k)
        f = fetch_s3(s3_client, k)
        lines = f.readlines()
        lines = map(str.strip, lines)
        lines = [ line for line in lines if line.startswith("<div>") ]
        print(lines, "\n\n\n\n\n")


if __name__ == "__main__":
    import sys
    import os
    os.environ['AWS_DEFAULT_REGION']='us-east-1'
    main(sys.argv)
