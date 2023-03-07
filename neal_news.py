#!/usr/bin/python3
# coding: utf-8
import boto3
import bs4
import datetime
import email, email.policy
import gzip
import io
import re
import urllib

BUCKET = 'www.neal.news'

INCOMING = 'neal.news.testing'


def parse_email(f,dump=False):
    print("parse_email")
    p = email.message_from_file(f, policy=email.policy.SMTPUTF8)

    frm, subj, dt = p.get("From"), p.get("Subject"), p.get("Date")
        
    body = p.get_body("html")
    html = body.get_payload(decode=True)

    if dump:
        with open('em.html', 'wb') as f:
            f.write(html)

    # Inline date
    dt = datetime.datetime.strptime(dt, "%a, %d %b %Y %H:%M:%S %z")
    dt = dt.strftime('%b %d, %Y')

    print("parse_email.bs4")
    return bs4.BeautifulSoup(html, 'html.parser'), dt

def unwrap_link(link):
    url = urllib.parse.urlparse(link)
    queries = urllib.parse.parse_qs(url.query)
    return {"href":  queries['url'][0]}

def clean_item(item):
    link, div = item
    link.attrs = unwrap_link(link['href'])
    link.span.unwrap()
    link.parent.unwrap()
    if link.contents[0]  == ' ':
        del link.contents[0]
    if link.contents[-1] == ' ':
        del link.contents[-1]

    publisher = div.find("div", {"itemprop":"publisher"}).span
    publisher.attrs = {}
    publisher.name = 'em'

    #strip all formatting from description
    desc = div.find("div", {"itemprop":"description"})
    desc.attrs = {}
    for t in desc.descendants: t.attrs = {}

    
    # strip all share buttons etc
    div.clear()
    div.insert(0, link)
    div.insert(1, publisher)
    div.insert(2, desc)


    return link.attrs["href"], div


def extract_items(soup):
    print("extract_items")

    items = map(clean_item,
                ((a, a.parent.parent) for a in soup.find_all("a",{"itemprop":"url"}))
                )
            

    uniq_href = set()
    uniq, i = [], -1

    for i, item in enumerate(items):
        href, item = item
        # removing protocol portion to eliminated dupe of http + https
        href = re.sub('^https?://', '', href)
        if href not in uniq_href:
            uniq_href.add(href)
            uniq.append(item)

#    print(uniq_href)

    print(f"{i+1} items, {len(uniq)} unique")

    return uniq

def build_new_index(items, d, yesterday_href):
    print("build_new_index")
    items = "\n    ".join(map(str, items))

    return f"""
    <!doctype html>
    <html>
    <head>
    <meta charset='UTF-8'/>
    <title>neal.news / {d}</title>
    <style>
    body {{
        max-width: 50rem;
        padding: 2rem;
        margin: auto;
        }}
    body > div {{
        margin: 1em
        }}
    a {{
        text-decoration: none
        }}
    </style>
    <base target="_blank">
    </head>
    <body>
    <h1>neal.news</h1>
    <h3>{d}</h3>
    {items}
    <a href="{yesterday_href}">yesterday's news</a>
    <script>
    document.onclick = function(e) {{
        a = e.composedPath().find((x) => x.href)
        if(!a) return;
        req = new XMLHttpRequest();
        req.open("POST", "https://jt5a7bev0m.execute-api.us-east-1.amazonaws.com/beta/")
        req.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
        req.send(a.href)
    }}
    document.onmouseup = e => e.which != 2 || document.onclick(e)
    </script>
    </body>
    </html>
    """

def update_yesterday(client):
    print("update_yesterday")
    # Method 2: Client.put_object()
    id = client.head_object(Bucket=BUCKET, Key='index.html')['ETag'][1:-1]
    yesterday_href = "%s.html" % id
    client.copy_object(
            CopySource={'Bucket': BUCKET,
                           'Key': 'index.html'},
            Bucket=BUCKET,
            Key=yesterday_href,
            ContentType='text/html',
            ContentEncoding='gzip' )
    return yesterday_href

def update_index(client, clean):
    print("update_index")
    client.put_object(
            Body=gzip.compress(str(clean).encode('UTF-8')),
            Bucket=BUCKET,
            Key='index.html',
            ContentType='text/html',
            ContentEncoding='gzip' )



def fetch_s3(client, id):
    print("fetch_s3")
    obj = client.get_object(Bucket=INCOMING, Key=id)
    f = io.StringIO(obj['Body'].read().decode('utf-8'))
    return f

def lambda_handler(event, context):
    client = boto3.client('s3')

    ses =  event['Records'][0]['ses'];
    print(f"handling {ses['mail']}")
    id =  ses['mail']['messageId']
    print(f"handling {id}")

    yesterday_href = update_yesterday(client)

    f = fetch_s3(client, "alerts/"+id)
    soup, dt = parse_email(f)
    items = extract_items(soup)
    clean = build_new_index(items, dt, yesterday_href)

    update_index(client, clean)
    
    try: 
        sagemaker()
    except:
        pass

def sagemaker():
    client = boto3.client("sagemaker")
    now = datetime.datetime.now()
    
    score_job = client.create_training_job(
        TrainingJobName="neal-news-score-%d" % int(now.timestamp()),
        AlgorithmSpecification={
            'TrainingImage': '887983324737.dkr.ecr.us-east-1.amazonaws.com/neal-news:latest',
            'TrainingInputMode': 'File'
        },
        RoleArn='arn:aws:iam::887983324737:role/service-role/AmazonSageMaker-ExecutionRole-20200125T123071',
        HyperParameters={
            'NEWS_MODE': 'score_update'
        },
#        InputDataConfig=[
#        ],
        OutputDataConfig={
            'S3OutputPath': 's3://njnmdummy2/'
        },
        ResourceConfig={
            'InstanceType': 'ml.g4dn.xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 5,
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 3600,
            'MaxWaitTimeInSeconds':4600
        },
        EnableNetworkIsolation=False,
        EnableManagedSpotTraining=True,
    )    
    
    
if __name__ == '__main__' :
    import sys
    with open(sys.argv[1]) as f:
        soup, d = parse_email(f, True)
    items = extract_items(soup)
    clean = build_new_index(items, d, "#not_implemented")
    with open('index.html', 'w') as f:
        f.write(str(clean))
