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
    
    first_ts = datetime.datetime.strptime(first_ts, "%Y-%m-%d %H:%M:%S.%f")
    return clicks, first_ts



def main(args):
    clicks, first_ts =  get_logs()
    print(clicks)
    print(first_ts)

if __name__ == "__main__":
    import sys
    import os
    os.environ['AWS_DEFAULT_REGION']='us-east-1'
    main(sys.argv)
