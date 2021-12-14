import pyterrier as pt
import json
import pandas as pd
import os
import shutil
import lightgbm as lgb
from sklearn.model_selection import train_test_split
def rm_r(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

def get_top_10_related(query):
    if not pt.started():
        pt.init()
    rm_r("pd_index")
    pd_indexer = pt.DFIndexer("./pd_index")
    df = pd.read_csv("index_file.csv",index_col=0)
    df = pd.read_csv("index_file.csv",index_col=0)
    df["docno"] = df["docno"].astype("str")
    df["comment"] = df["comment"].astype("str")
    df["dislike"] = df["dislike"].astype("str")
    df["favorite"] = df["favorite"].astype("str")
    df["like"] = df["like"].astype("str")
    df["view"] = df["view"].astype("str")
    df["per_like_dislike"] = df["per_like_dislike"].astype("str")
    a = dict(zip(df.docno,df.id))
    # load index
    indexref = pd_indexer.index(df["text"],df["docno"],df["tags"],df["comment"],df["dislike"],df["like"],df["view"],df["per_like_dislike"])

    # pipeline aggregate

    tf = pt.BatchRetrieve(indexref, wmodel="Tf")
    tfidf = pt.BatchRetrieve(indexref, wmodel="TF_IDF")
    bm25 = pt.BatchRetrieve(indexref, wmodel="BM25")
    qe = pt.rewrite.Bo1QueryExpansion(indexref)

    # Some Added On Feature
    # score top 75% view
    top_75_view = df["view"].astype("float").describe()["75%"]
    top_75_view
    # top 75% like/dislike
    top_75_like_dislike = df["per_like_dislike"].astype("float").describe()["75%"]
    top_75_like_dislike

    pipe_qe = bm25 >> qe >> bm25

    ltr_feats = (pipe_qe % 100) >> pt.text.get_text(indexref, ["docno", "tags", "dislike", "like", "view", "per_like_dislike"]) >> (

        pt.transformer.IdentityTransformer()
        ** # score of text for query
        (pt.apply.query(lambda row: query) >> tfidf)
        ** # TF-IDF score for tag
        (pt.text.scorer(body_attr="tags", wmodel="TF_IDF") )
        ** # view more than 100
        (pt.apply.doc_score(lambda row: float(row["view"]) > top_75_view) )
        ** # top 75 like/dislike
        (pt.apply.doc_score(lambda row: float(row["view"]) > top_75_like_dislike) )
    )
    # this configures LightGBM as LambdaMART
    lmart_l = lgb.LGBMRanker(
        task="train",
        silent=False,
        min_data_in_leaf=1,
        min_sum_hessian_in_leaf=1,
        max_bin=255,
        num_leaves=31,
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10],
        ndcg_at=[10],
        eval_at=[10],
        learning_rate= .1,
        importance_type="gain",
        num_iterations=100,
        early_stopping_rounds=5
    )
    df3_label = pd.read_csv("label.csv")
    df3_label['qid'] = df3_label['qid'].astype(str) 
    df3_label['docno'] = df3_label['docno'].astype(str)
    df3_label['label'] = df3_label['label'].astype(float)
    get_query = pd.read_csv("query.csv")

    train, test_topics =  train_test_split(get_query, test_size=0.15, random_state=40)
    train_topics, valid_topics =  train_test_split(train, test_size=0.3, random_state=40)
    lmart_x_pipe = ltr_feats >> pt.ltr.apply_learned_model(lmart_l, form="ltr", fit_kwargs={'eval_at':[10]})
    lmart_x_pipe.fit(train_topics, df3_label, valid_topics, df3_label)
    # return results
    results = lmart_x_pipe.search(query)["docid"].tolist()[:10]
    return results

def get_id_title(docno):
    data = pd.read_csv("id_map.csv",index_col=0)
    data_doc = data.set_index("docno").T
    return data_doc[docno]["id"], data_doc[docno]["title"]