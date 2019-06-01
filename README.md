# Alerts -> Static Site convertor

This is a small project to aggregate news feeds for my personal use.

## Environment

(http://neal.news) is hosted on AWS. A rough outline is:

  1. Email recieved by SES, stored to S3
  2. Lambda function fetches email, extracts content,
     apply's template, pushes HTML back to S3.
  3. S3 serves static site.
