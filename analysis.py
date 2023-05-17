#!/usr/bin/python3

import boto3
import collections
import datetime
import io
import gzip
import pickle
import time
import re

import neal_news

from bs4 import BeautifulSoup

MSG_PREFIX = '(xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx) Method request body before transformations: '
BUCKET = 'www.neal.news'

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
    
    first_ts = datetime.datetime.strptime(first_ts, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc)
    return clicks, first_ts


def get_docs_keys(client, oldest):

    listing = client.list_objects_v2(Bucket=BUCKET)

    for i in listing['Contents']:
        if i['LastModified'] > oldest and i['Key'].endswith('.html'):
            yield i['Key']

# Different - ungzips the data first!
def fetch_s3(client, id):
    #print("fetch_s3")
    obj = client.get_object(Bucket=BUCKET, Key=id)
    f = io.StringIO(gzip.open(obj['Body']).read().decode('utf-8'))
    return f, obj

def date_to_features(obj, delta=1):
    import datetime
    d = obj['LastModified']
    d = d - datetime.timedelta(days=delta) # Modification - 1 day, bc old days are rotated when new one comes in 

    return d.weekday(), d.timetuple().tm_yday

pat_url = re.compile('https?://[^"]*')
pat_txt = re.compile("</?[^>]*>")


def get_lines(s3_client, k, j):
    f, obj = fetch_s3(s3_client, k)
#    lines = f.readlines()
#    lines = map(str.strip, lines)
#    lines = [ line for line in lines if line.startswith("<div") ]

    soup = BeautifulSoup(f, 'html.parser')
    lines = lines = soup.select("body > div")


    wday, yday = date_to_features(obj, 1 - int(k == 'index.html'))
    # print("*" + str(len(lines)))

    ret = []
    for i, line in enumerate(lines):
        # print("***\n" + line + "\n\n\n")


        ret.append (( 
                        line.a.attrs['href'],
                        line,
                        wday,
                        yday,
                        i,
                        j,
                        len(lines)
                    ))
    return ret, [str(l) for l in lines] 


def get_files(doc_keys=None, drop=True):
    clicks, first_ts = get_logs()
    print(first_ts)
    print(len(clicks))
    s3_client = boto3.client('s3')
    if doc_keys is None:
        doc_keys = get_docs_keys(s3_client, first_ts + datetime.timedelta(-1))

    X = list()

    for j, k in enumerate(doc_keys):
        #print(k)

        lines, orig = get_lines(s3_client, k, j)

        for line in lines:
            if line[0] in clicks:
                X = X + lines
                break
        else:
            if not drop:
                X = X + lines
            else:
                print('No clicks found in ' + k)

    url, X, wday, yday, i, j, n = zip(*X)
    Y = [int(x in clicks) for x in url]

    return Y, X, wday, yday, i, j, n, orig


def gen_embedding(lines, model, tokenizer, transform, ctx):
    return [gen_embeddings_line(line, model, tokenizer, transform, ctx) for line in lines]


def gen_embeddings_line(line, model, tokenizer, transform, ctx):
    import numpy
    import mxnet as mx

    tags = []
    sentence = ""

    for _, i in enumerate(line.find_all(string=True)):
        toks = tokenizer(str(i))
        tags.append((toks, list(x.name for x in i.parents)))
        sentence = sentence + " " + str(i)


    sample = transform([sentence]);
    words, valid_len, segments = mx.nd.array([sample[0]]).as_in_context(ctx), mx.nd.array([sample[1]]).as_in_context(ctx), mx.nd.array([sample[2]]).as_in_context(ctx);
    seq_encoding, cls_encoding = model(words, segments, valid_len)

    seq_encoding = seq_encoding[0,:,:].asnumpy()
    
    ### Post processing to deal with BPE
    t_iter = iter(tags)

    out = list()
    bpe_cnt = -9

    toks, parent, t_i = (*next(t_iter), 0)
    for i in range(int(valid_len.asscalar())):

        token_id = words[0][i].asscalar()
        if token_id == 1:
            # [PAD] token, sequence is finished.
            break
        if (token_id in (2, 3)):
            # [CLS], [SEP]
            #print("skip")
            continue



        while(len(toks) <= t_i):
            toks, parent, t_i = (*next(t_iter), 0)
            #print("chug")
        #print(t_i, i, len(toks), toks[t_i])

        if toks[t_i].startswith('##'):
            #print(toks[t_i - 1], toks[t_i])
            out[-1][0] += toks[t_i][2:]
            out[-1][1] += seq_encoding[i,:]
            bpe_cnt += 1
        else:
            if bpe_cnt > 1:
                out[-1][1] /= bpe_cnt
                #print(bpe_cnt, out[-1][0])
            bpe_cnt = 1

            P = [
                1 if 'a' in parent else 0,
                1 if 'b' in parent else 0,
                1 if 'em' in parent else 0,
            ]


            out.append([ toks[t_i], seq_encoding[i,:], P])

        t_i = t_i + 1

    TOKENS, X, P = zip(*out)
    X = numpy.vstack(X)
    P = numpy.array(P)
    return TOKENS, X, P

def gen_features(X, wday, yday, i, j, n, tf=None, u=None, n_features=1400):
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.decomposition import TruncatedSVD
    from scipy.spatial.distance import cosine, cdist

    from collections import Counter
    import numpy

#     from bert_embedding import BertEmbedding

#     import mxnet as mx

#     ctx = mx.gpu(0)
#     bert_embedding = BertEmbedding(ctx=ctx)
    
#     result = bert_embedding(X)

    import gluonnlp as nlp;
    import mxnet as mx;

    print("Loading BERT to gpu")
    
    ctx = mx.gpu()

    model, vocab = nlp.model.get_model('bert_24_1024_16',
                                       dataset_name='book_corpus_wiki_en_uncased',
                                       ctx=ctx, use_classifier=False, use_decoder=False)
    tokenizer = nlp.data.BERTTokenizer(vocab, lower=True)
    transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=128, pair=False, pad=False)

    print("Creating BERT embeddings")

    result = gen_embedding(X, model, tokenizer, transform, ctx)
    
    
    
    del model, tokenizer, transform
    ctx.empty_cache()
    
    if tf is None:
        print("Term Freq:", len(result))

        tf = Counter()
        for r in result:
            tf.update(r[0])

    N = sum(tf.values())

    h = FeatureHasher(n_features=n_features, input_type="string")

    def s_from_w(s):
        words, embedding, P = s
        embedding = numpy.concatenate((embedding, h.transform([w] for w in words).toarray(), P), axis=1)
        weight = numpy.array([1/(1+tf[x]/N)/len(words) for x in words])
        return weight.dot(embedding)

    print("Augmenting with TF/IDF")
    SX = numpy.array([s_from_w(x) for x in result])

    if u is None:
        print("SVD:", SX.shape)
        svd = TruncatedSVD(n_components=1, n_iter=16, random_state=42)
        svd.fit(SX)
        print(svd.explained_variance_ratio_)
        u = svd.components_

    v2 = SX - SX.dot(u.transpose())*u

    wday = numpy.array(wday, ndmin=2)
    yday = numpy.array(yday, ndmin=2)

    max_sim = wday * 0

    i = numpy.array(i)
    j = numpy.array(j)

    print("Similarity calculation")
    for K, _ in enumerate(max_sim):
        if i[K] > 0 :
            max_sim[K] = (1-cdist(SX[[K],:], SX[ (j == j[K]) & (i < i[K]), :], metric='cosine')**2).max()


    i = numpy.array(i, ndmin=2)

    i_scaled = i / numpy.array(n, ndmin=2)

    return numpy.hstack((wday.T, yday.T, i.T, i_scaled.T, max_sim.T, v2)), tf, u


def train(time_allowed=20, trials=None, output="model.pickle") :
    import xgboost as xgb
    import numpy
    from hyperopt import hp, tpe, Trials
    from hyperopt.fmin import fmin

    Y, *R, _ = get_files()
    X, tf, u = gen_features(*R)
    del R

    print(f"{X.shape} cases")
    
    dtrain = xgb.DMatrix(X, Y)

    if trials is None:
        trials = Trials()

    param = {'verbosity':0, 'objective':'binary:logistic', 'tree_method':'gpu_hist', "predictor":'gpu_predictor'}
    num_round = 128
    #nfold = 7

    def ar_m_to_log_m(Ex, Vx):
        mu = 2*numpy.log(Ex) - .5*numpy.log(Vx + Ex*Ex)
        sigma2 = numpy.log1p(Vx/Ex/Ex)
        mode = numpy.exp(mu - sigma2)
        return mode
    
    def objective(params):
        params['max_depth'] = int(params['max_depth'])

        params.update(param)
        seed = int(params.pop('seed'))
        nfold = int(params.pop('nfold'))


        result = xgb.cv(params, dtrain, num_round, nfold=nfold, metrics={'auc'}, seed=seed)

        # is this no longer an array? Can be list if nan is present?
        score = max(ar_m_to_log_m(m, s*s) for m,s in zip(result['test-auc-mean'], result['test-auc-std']))

        return -score

    space = {
        'max_depth': hp.quniform('max_depth', 2, 10, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
        'gamma': hp.uniform('gamma', 0.01, 0.5),
        'subsample': hp.uniform('subsample', .3, 1),
        'scale_pos_weight': hp.uniform('scale_pos_weight', .8, 20.0),
        'eta': hp.uniform('eta', .01, .4),
        'seed': hp.randint('seed', 4),
        'nfold': hp.quniform('nfold', 5,8,1),
    }



    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=len(trials.trials) + 25)

    param.update(best)
    param['max_depth'] = int(param['max_depth']) #fixme
    seed = int(param.pop('seed')) #fixme
    nfold = int(param.pop('nfold')) #fixme

    
    print("re-cv to find early stopping")
    cv = xgb.cv(param, dtrain, num_round, nfold=nfold, metrics={'auc'}, seed=seed)

    #num_round = (r['test-auc-mean'] - r['test-auc-std']/2).idxmax()
    
    
    num_round = numpy.array([ar_m_to_log_m(m, s*s) for m,s in zip(cv['test-auc-mean'], cv['test-auc-std'])])
    num_round = num_round - numpy.linspace(0, num_round.shape[0]*.0003, num=num_round.shape[0])
    num_round = num_round.argmax() + 1

    print(f"final train with {num_round}")
    model = xgb.train(param, dtrain, num_round)

    MODEL = (model, param, trials, tf, u)

    if output:
        print("pickle to s3")
        client = boto3.client('s3')
        client.put_object(
                Body=gzip.compress(pickle.dumps(MODEL)),
                Bucket=BUCKET,
                Key=output,
                ContentType='application/python-pickle',
                ContentEncoding='gzip' )


    #with open("model.pickle", "wb") as f:
    #    pickle.dump(MODEL, f)

    return MODEL


def score_index(model_key="model.pickle", save=True):
    import xgboost as xgb
    import numpy

    s3_client = boto3.client('s3')

    print("fetching most recent model")
    obj = s3_client.get_object(Bucket=BUCKET, Key=model_key)
    MODEL1 = pickle.load(gzip.open(obj['Body']))

    r, p, t, tf, u = MODEL1

    Y, *index, orig = get_files(['index.html'], drop=False)
    index[3] = [0 for _ in index[3]] # score as if all were in first slot.

    X, _, _ = gen_features(*index, tf=tf, u=u)

    # Remove links that were already clicked
    print("Removing %d already clicked links" % sum(Y))
    Y = numpy.array(Y)
    orig2 = [o for i,o in enumerate(orig) if Y[i] == 0]
    X = xgb.DMatrix(X[Y == 0, :], Y[Y == 0])

    yhat = r.predict(X)

    for i, _ in enumerate(yhat):
        orig2[i] = orig2[i].replace("<div", f"<div data-score0={yhat[i]} " ,1)
        # Five percent greedy-epsilon bandit
        if numpy.random.uniform() < .05 :
            yhat[i] = numpy.random.choice(yhat)
            #orig2[i] = orig2[i].replace("<div", "<div data-bandit=1", 1)

    # Rescore with positioning
    index2 = Y * 9999
    index2[Y == 0] =  numpy.argsort(yhat) # rescore per actual position.
    index[3] = index2
    X, _, _ = gen_features(*index, tf=tf, u=u)
    X = xgb.DMatrix(X[Y == 0, :], Y[Y == 0])
    yhat2 = r.predict(X)

    for i, _ in enumerate(yhat):
        orig2[i] = orig2[i].replace("<div", f"<div data-score1={yhat2[i]} " ,1)
        # Five percent greedy-epsilon bandit
        if numpy.random.uniform() < .05 :
            yhat2[i] = numpy.random.choice(yhat2)
            orig2[i] = orig2[i].replace("<div", "<div data-bandit=1 ", 1)

    scores, lines = list(zip(*sorted(zip(-(yhat+yhat2)/2, orig2))))

    body, _, = fetch_s3(s3_client, "index.html")
    body = "".join(body.readlines())

    d = re.search("(?<=<h3>).*(?=</h3>)", str(body)).group(0)

    yesterdays_href = re.search('(?<=<a href=")[0-9a-f]*[.]html(?=">yesterday\'s news</a>)', body).group(0)

    new_index = neal_news.build_new_index(lines, d, yesterdays_href)

    if save:
        neal_news.update_index(s3_client, new_index)



def main(news_mode):
    if news_mode == "train":
        train()
    elif news_mode == "score":
        score_index()
    elif news_mode == "score_update":
        score_index()
        if datetime.datetime.now().weekday() == 0:
            train()
        
if __name__ == "__main__":
    print("Starting analysis.py")
    import json
    with open('/opt/ml/input/config/hyperparameters.json') as f:
        hyper = json.load(f)
    print(hyper)
    main(hyper.get("NEWS_MODE", "score"))
