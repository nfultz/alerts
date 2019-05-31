#!/usr/bin/python3
# coding: utf-8
import boto3
import bs4
import datetime
import email, email.policy
import gzip
import io
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

    return bs4.BeautifulSoup(html, 'html.parser'), dt

def unwrap_link(link):
    url = urllib.parse.urlparse(link)
    queries = urllib.parse.parse_qs(url.query)
    return {"href":  queries['url'][0], "target":"_blank" }

def clean_item(item):
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
    if desc.div.string:
        desc.div.string = desc.div.string.strip()
    desc.div.name = 'em'

    # strip all formatting
    desc.attrs = {}
    for t in desc.descendants: t.attrs = {}

    desc.insert(0, link)
    return desc


def extract_items(soup):
    print("extract_items")

    return map(clean_item,
               ((tr.div.a, tr.div.div.div)
                   for tr in soup.find_all("tr", itemtype="http://schema.org/Article")
                   if tr.div.a))

def build_new_index(items, d, yesterday_href):
    print("build_new_index")

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
    </head>
    <body>
    <h1>neal.news</h1>
    <h3>{d}</h3>
    {"".join(map(str, items))}
    <a href="{yesterday_href}">yesterday's news</a>
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

if __name__ == '__main__' :
    import sys
    with open(sys.argv[1]) as f:
        soup, d = parse_email(f, True)
    items = extract_items(soup)
    clean = build_new_index(items, d, "#not_implemented")
    with open('index.html', 'w') as f:
        f.write(str(clean))
