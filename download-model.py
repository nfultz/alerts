import urllib.request, zipfile, io, os


try:
    print("making directories")
    os.mkdir("/root/.mxnet")
    os.mkdir("/root/.mxnet/models")
except:
    pass

print("download vocab")
f = urllib.request.urlopen("https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/vocab/book_corpus_wiki_en_uncased-a6607397.zip")
with zipfile.ZipFile(io.BytesIO(f.read())) as myzip:
    myzip.extract('book_corpus_wiki_en_uncased-a6607397.vocab', '/root/.mxnet/models/')

print("download params")
# f = urllib.request.urlopen("https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/bert_24_1024_16_book_corpus_wiki_en_uncased-75cc780f.zip")
# with zipfile.ZipFile(io.BytesIO(f.read())) as myzip:
#     myzip.extract('bert_12_768_12_book_corpus_wiki_en_uncased-75cc780f.params', '/root/.mxnet/models/')
    
    
b1024 = "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/bert_24_1024_16_book_corpus_wiki_en_uncased-24551e14.zip"    

f = urllib.request.urlopen(b1024)
with zipfile.ZipFile(io.BytesIO(f.read())) as myzip:
    myzip.extract('bert_24_1024_16_book_corpus_wiki_en_uncased-24551e14.params', '/root/.mxnet/models/')
