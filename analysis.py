#!/usr/bin/python3

import boto3
import collections
import datetime
import time

MSG_PREFIX = '(xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx) Method request body before transformations: '

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
    import pprint
    while status not in ('Complete', 'Failed','Cancelled', 'Timeout'):
        time.sleep(10)
        results = client.get_query_results(queryId=start['queryId'])
        status = results['status']


    clicks = collections.defaultdict(dict)
    for i, record in enumerate(results['results']):
        for field in record:
            clicks[i][field['field']] = field['value']
        clicks[i] = clicks[i]['message'][len[MSG_PREFIX):]
    
    print(pprint.pprint(clicks))




def main(args):
    get_logs()

if __name__ == "__main__":
    import sys
    import os
    os.environ['AWS_DEFAULT_REGION']='us-east-1'
    main(sys.argv)
