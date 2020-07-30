# Google Alerts -> Static Site convertor

This is a small project to aggregate news feeds (via Google Alerts) for my personal use at [neal.news](http://neal.news).

## Environment

neal.news is hosted on AWS. A rough outline is:

  1. Email recieved by SES, stored to S3
  2. Lambda function fetches email, extracts content,
     applys template, pushes HTML back to S3, and queues a scoring job.
  3. S3 serves static site.

### Manually firing a lambda job from AWS UI

* Go to create test event
* Fill in the `messageId` below - it is the only field actually used, the other mail headers will be pulled from the file with that ID on S3
```
{
  "Records": [
    {
      "eventSource": "aws:ses",
      "eventVersion": "1.0",
      "ses": {
        "mail": {
          "timestamp": "2020-07-30T14:20:54.877Z",
          "source": "3RdciXxQKAH4iqqingcngtvu-pqtgrn0iqqing.eqo@alerts.bounces.google.com",
          "messageId": "qlkjb610bfq4c4nlm99u2plkh9li82uo4iu2s401",
          "destination": [
            "foo@neal.news"
          ]
        }
      }
    }
  ]
}
```

## Item Ranking / Scoring

  1. Clicked items are logged back to CloudWatch events (via an API Gateway).
  2. Those historical clicks are used to train a classifier
  
      1. Currently using BERT + xgboost
      2. Retrained on a SageMaker GPU spot instance every Monday.
 
 3. That model is used to re-rank incoming news.

      1. Read/write index.html from S3.
      2. 10% of items are scored randomly to help mitigate overfitting.

