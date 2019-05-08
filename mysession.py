# coding: utf-8
import boto3
import bs4
import datetime
import email
import email.policy
import gzip
import uuid
import urllib

with open("Google_Alert_-_Daily_Digest_4.eml") as f:
    p = email.message_from_file(f, policy=email.policy.SMTPUTF8)

frm, subj = p.get("From"), p.get("Subject")
    
e = p.get_body("html")
html = e.get_payload(decode=True)
with open('em.html', 'wb') as f: f.write(html )

soup = bs4.BeautifulSoup(html, 'html.parser')


items = [(tr.div.a, tr.div.div.div) for tr in soup.find_all("tr", itemtype="http://schema.org/Article") if tr.div.a ]


def unwrap_link(href):
    url = urllib.parse.urlparse(link['href'])
    queries = urllib.parse.parse_qs(url.query)
    return {"href":  queries['url'][0] }


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
""")

#clean = bs4.BeautifulSoup(requests.get("http://neal.news").content)

# Inline date
d = p.get("Date")
d = datetime.datetime.strptime(d, "%a, %d %b %Y %H:%M:%S %z")
d = d.strftime('%b %d, %Y')

clean.head.title.string = "neal.news / " + d

h3 = soup.new_tag("h3")

h3.string = d

yesterday = soup.new_tag("a", href = "%s.html" % uuid.uuid1())
yesterday.string = "yesterday's news"

clean.body.h1.insert_after(h3, *items, yesterday)


with open("index.html", 'w') as f: f.write(str(clean))


# Method 2: Client.put_object()
client = boto3.client('s3')
client.copy_object(
        CopySource={'Bucket': 'www.neal.news', 'Key': 'index.html'},
        Bucket='www.neal.news',
        Key=yesterday['href'],
        ContentType='text/html',
        ContentEncoding='gzip' )

client.put_object(
        Body=gzip.compress(str(clean).encode('UTF-8')),
        Bucket='www.neal.news',
        Key='index.html',
        ContentType='text/html',
        ContentEncoding='gzip' )
