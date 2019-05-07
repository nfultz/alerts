# coding: utf-8
import boto3
import bs4
import datetime
import email
import email.policy
import gzip
import urllib

with open("Google_Alert_-_Daily_Digest_3.eml") as f:
    p = email.message_from_file(f, policy=email.policy.SMTPUTF8)

frm, subj = p.get("From"), p.get("Subject")
    
e = p.get_body("html")
html = e.get_payload(decode=True)
with open('em.html', 'wb') as f: f.write(html )

soup = bs4.BeautifulSoup(html, 'html.parser')


items = [(tr.div.a, tr.div.div.div) for tr in soup.find_all("tr", itemtype="http://schema.org/Article") if tr.div.a ]


for link, desc in items:
    link.attrs = {"href":  urllib.parse.parse_qs(urllib.parse.urlparse(link['href']).query)['url'][0] }
    link.span.unwrap()
    if link.contents[0] == ' ': 
        del link.contents[0]
    if link.contents[-1] == ' ': 
        del link.contents[-1]
    desc.div.a.span.unwrap()
    desc.div.a.unwrap()
    desc.div.name = 'em'
    for t in desc.descendants: t.attrs = {}
    desc.insert(0, link)
    desc.attrs = {}

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
</html>""")

#clean = bs4.BeautifulSoup(requests.get("http://neal.news").content)

# Inline date
d = p.get("Date")
d = datetime.datetime.strptime(d, "%a, %d %b %Y %H:%M:%S %z").strftime('%b %d, %Y')

clean.head.title.string = "neal.news / " + d

h3 = soup.new_tag("h3")

h3.string = d

clean.body.h1.insert_after(h3)
last = clean.body.h3
for _, div in items:
    last.insert_after(div)
    last = last.next_sibling



with open("index.html", 'w') as f: f.write(str(clean))


# Method 2: Client.put_object()
client = boto3.client('s3')
client.put_object(
        Body=gzip.compress(str(clean).encode('UTF-8')),
        Bucket='www.neal.news',
        Key='index.html',
        ContentType='text/html',
        ContentEncoding='gzip' )
