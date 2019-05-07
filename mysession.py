# coding: utf-8
import datetime
import email
import email.policy
with open("Google_Alert_-_Daily_Digest_1.eml") as f:
    p = email.message_from_file(f, policy=email.policy.SMTPUTF8)
    
e = p.get_body("html")
html = e.get_payload(decode=True)
with open('em.html', 'wb') as f: f.write(html )

import bs4
import urllib
soup = bs4.BeautifulSoup(html, 'html.parser')


items = [(tr.div.a, tr.div.div.div) for tr in soup.find_all("tr") if tr.get('itemtype') == "http://schema.org/Article" and tr.div.a ]


for i in items:
    i[0].attrs = {"href":  urllib.parse.parse_qs(urllib.parse.urlparse(i[0]['href']).query)['url'][0] }
    i[0].span.unwrap()
    i[1].div.a.span.unwrap()
    i[1].div.a.unwrap()
    i[1].div.name = 'em'
    for t in i[1].descendants: t.attrs = {}
    i[1].insert(0, i[0])
    i[1].attrs = {}

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
<h3></h3>
</body>
</html>""")

for i in items:
    clean.append(i[1])

# Inline date
d = p.get("Date")
d = datetime.datetime.strptime(d, "%a, %d %b %Y %H:%M:%S %z").strftime('%b %d, %Y')

clean.head.title.string = "neal.news / " + d
clean.body.h3.string = d

with open("index.html", 'w') as f: f.write(clean.prettify())

import boto3

# Method 2: Client.put_object()
client = boto3.client('s3')
client.put_object(Body=clean.prettify(), Bucket='www.neal.news', Key='index.html', ContentType='text/html')
