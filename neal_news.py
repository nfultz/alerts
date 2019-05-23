#!/usr/bin/python3
# coding: utf-8
import boto3
import bs4
import datetime
import email, email.policy
import gzip
import io
import uuid
import urllib

BUCKET = 'www.neal.news'

INCOMING = 'neal.news.testing'

def parse_email(f,dump=False):
    print("parse_email")
    p = email.message_from_file(f, policy=email.policy.SMTPUTF8)

    frm, subj = p.get("From"), p.get("Subject")
        
    e = p.get_body("html")
    html = e.get_payload(decode=True)

    if dump:
        with open('em.html', 'wb') as f:
            f.write(html)

    # Inline date
    d = p.get("Date")
    d = datetime.datetime.strptime(d, "%a, %d %b %Y %H:%M:%S %z")
    d = d.strftime('%b %d, %Y')

    return bs4.BeautifulSoup(html, 'html.parser'), d

def unwrap_link(link):
    url = urllib.parse.urlparse(link)
    queries = urllib.parse.parse_qs(url.query)
    return {"href":  queries['url'][0], "target":"_blank" }

def extract_items(soup):
    print("extract_items")

    items = [(tr.div.a, tr.div.div.div) for tr in soup.find_all("tr", itemtype="http://schema.org/Article") if tr.div.a ]


    for i, item in enumerate(items):
        link, desc = item
        link.attrs = unwrap_link(link['href'])
        link.span.unwrap()
        if link.contents[0]  == ' ':
            del link.contents[0]
        if link.contents[-1] == ' ':
            del link.contents[-1]

        #change publisher div to em
        desc.div.a.span.unwrap()
        desc.div.a.unwrap()
        desc.div.name = 'em'

        # strip all formatting
        desc.attrs = {}
        for t in desc.descendants: t.attrs = {}

        desc.insert(0, link)
        items[i] = desc

    return items
        

def build_new_index(items, d):
    print("build_new_index")

    clean = bs4.BeautifulSoup("""
    <!doctype html>
    <html>
    <head>
    <meta charset='UTF-8'/>
    <title></title>
    <style>
    body {
        max-width: 50rem;
        padding: 2rem;
        margin: auto;
        }
    body > div {
        margin: 1em
        }
    </style>
    </head>
    <body>
    <h1>neal.news</h1>
    </body>
    </html>
    """, 'html.parser')

    #clean = bs4.BeautifulSoup(requests.get("http://neal.news").content)


    clean.head.title.string = "neal.news / " + d

    h3 = clean.new_tag("h3")
    h3.string = d

    yesterday = clean.new_tag("a", href = "%s.html" % uuid.uuid1())
    yesterday.string = "yesterday's news"

    succ = clean.body.h1
    for i in [h3, *items, yesterday]:
      succ.insert_after(i)
      succ = i

    return clean, yesterday['href']


def update_s3(clean, old):
    print("update_s3")
    # Method 2: Client.put_object()
    client = boto3.client('s3')
    print("update_s3.rotate")
    client.copy_object(
            CopySource={'Bucket': BUCKET,
                           'Key': 'index.html'},
            Bucket=BUCKET,
            Key=old,
            ContentType='text/html',
            ContentEncoding='gzip' )

    print("update_s3.refresh")
    client.put_object(
            Body=gzip.compress(str(clean).encode('UTF-8')),
            Bucket=BUCKET,
            Key='index.html',
            ContentType='text/html',
            ContentEncoding='gzip' )



def fetch_s3(id):
    print("fetch_s3")
    client = boto3.client('s3')
    obj = client.get_object(Bucket=INCOMING, Key=id)
    f = io.StringIO(obj['Body'].read().decode('utf-8'))
    return f

def lambda_handler(event, context):
    ses =  event['Records'][0]['ses'];
    print(f"handling {ses['mail']}")
    id =  ses['mail']['messageId']
    print(f"handling {id}")

    f = fetch_s3("alerts/"+id)
    soup, d = parse_email(f)
    items = extract_items(soup)
    clean, yesterday_href = build_new_index(items, d)
    update_s3(clean, yesterday_href)

if __name__ == '__main__' :
    import sys
    with open(sys.argv[1]) as f:
        soup, d = parse_email(f, True)
    items = extract_items(soup)
    clean, yesterday_href = build_new_index(items, d)
    with open('index.html', 'w') as f:
        f.write(str(clean))
