# YouTube-Video-Search-Engine
[flask.png]
## Introduction
With the increasing demand for electronic devices, it is hard for people to choose the best products from multiple brands. In this case, the unboxing video will be useful letting people get an overview of the product. We would like to build a basic search engine for video on Youtube based on the speech content transcript to try to improve the searching result. The goal of the project is to find the most relevant video according to the content for the web search engine.

## Data
To build our own datasets, we start from generate a unique query list related to the electronic field. The next step is to use the query to scrape some basic informations about the video on YouTube. In this step, we used the YouTube API key which is easy to apply.
To enlarge our datasets, we decides to include more textual features such as the transcripts and the tags of the video and some numeric features.
Other than that, we decides to analyze the comment under each video to get a specific view from the public. As we know, the comment section usually contains the most real evaluation of the video. To use this valuable information, we decides to make some sentiment analysis to help to evaluate the video itself from a new aspects. After comparing different methods online, we choose to use the Flair library which is time saving with good performance on the result to evaluate the top 100 popular comments for each video.
Since our dataset contains around 5700 videos in total, we decides to annotate 2000 videos manually to help to evaluate our model in later process. The annotation label is int in the range from 0 to 5 where 5 represents matched results.
## Methods
In our project, we used the PyTerrier as our main tools to build the baseline and our model.
### Baseline Model
The baseline model we build is BM25 with its default parameter. BM 25 stands for Best Match 25, it ranks a set of documents based on the query terms appearing in each document, regardless of their proximity within the document.
### Query Expansion
Query expansion is an effective way to improve the performance of retrieval. The idea of query expansion is to find the similarity between the query and the tags and append the most related tags into the query to search for results. A detailed query may lead to more accurate results. Many search engines like Google would suggest related queries in response to a query and then opt to use one of these alternative query suggestions. There are two types of query expansion methods Q > Q and R > Q. We used Bo1QueryExpansion which is a method of rewriting a query by making use of an associated set of documents. In our model, the pipeline takes the results from the BM25 and adds more query words based on the occurrences of terms in the BM25 result, and then retrieves the results using BM25 again.
Learn to Rank
Learning to rank is another method to improve performance. It is an algorithmic technique that applies supervised machine learning to solve ranking problems in the search engine. Before implementing learn to rank, we need to define features like other machine learning tasks first. According to our data exploration, we decided to add five features.
1. BM25 — query expansion score
2. TF-IDF retrieval score
3. Does the video tag include query, scored by TF-IDF
4. Whether the number of view in top 25%
5. Whether the ratio of like and dislike in top25%

We built four learn to rank models including coordinate ascent, random forests, SVM and LambdaMART. We want to evaluate these four models using MAP and NDCG, then choose the one that has the best performance. The idea of Coordinate ascent is optimizing through minimization of measure-specific loss, MAP in this case. LambdaMART is like a method of boosted regression tree in the area of information retrieval and is popular in the industry. We split the data to train and test the dataset, we trained the models based on training data and evaluated them based on the test dataset.
## Evaluation and Results
To evaluate the result, we used Mean Average Precision (MAP) and (Normalized Discounted Cumulative Gain (NDCG) scores for each model. MAP determines average precision for each query, then averages over queries, and NDCG solves the problem when comparing a search engine’s performance from one query to the next. From the chart below, we can notice that both MAP and NDCG of learning to rank models improves a lot compared to the baseline model.

### Feature Importance
In our project, we used different methods to calculate the feature importance and got a similar result.
To successfully figure out the correlation between each feature to the final label score, we calculated the correlation matrix and used heatmap to show the result. The result shows us that the rating score which is the label score is most related to the tags feature we build in our pipeline. Then, we may want to give the tags feature a higher weight in our model.

### Simple Ranking Function
We also build a simple ranking function model to improve the performance of the ranking result. The idea is to give different weights to different features. We also included the hyper-parameter to help to tune the model. After doing the feature importance analysis, we build the model r below where r1 represents the score return by the baseline model BM25 and r2 return by another baseline model TF-IDF.

Ranking function.
The results show us an improvement on our models but since we are training on a small dataset, the increase of NDCG score does not mean the model is good enough to rank.

NDCG score before and after applying the ranking function.
Feature Works
In our project, the datasets are constrained to certain fields of datas and this may influence the performance of models. We may want to explore our datasets by containing different fields to build a more general search engine. Also, even though we build some of the features, most of them are rarely important in our model. In later work, we may want to get more familiar with our data and to extract more features.

## Deliverable
The final deliverable is a Flask App, the user could enter the query related to the electronic devices, for example “new iphone 13”, “DJI drone”, the system would use our best model to retrieve the results and show the top10 most relevant Youtube Video title and URL, and then user could click URL to watch the video.
