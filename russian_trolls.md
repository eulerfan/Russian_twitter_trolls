Russian Twitter Trolls
================

Context

As part of the House Intelligence Committee investigation into how Russia may have influenced the 2016 US Election, Twitter released the screen names of almost 3000 Twitter accounts believed to be connected to Russia’s Internet Research Agency, a company known for operating social media troll accounts. Twitter immediately suspended these accounts, deleting their data from Twitter.com and the Twitter API. A team at NBC News including Ben Popken and EJ Fox was able to reconstruct a dataset consisting of a subset of the deleted data for their investigation and were able to show how these troll accounts went on attack during key election moments. This dataset is the body of this open-sourced reconstruction.

For more background, read the NBC news article publicizing the release: "Twitter deleted 200,000 Russian troll tweets. Read them here."[NBC Russian Tweets](https://www.nbcnews.com/tech/social-media/now-available-more-200-000-deleted-russian-troll-tweets-n844731)

Content This dataset contains two CSV files. tweets.csv includes details on individual tweets, while users.csv includes details on individual accounts.

To recreate a link to an individual tweet found in the dataset, replace user\_key in <https://twitter.com/user_key/status/tweet_id> with the screen-name from the user\_key field and tweet\_id with the number in the tweet\_id field.

Following the links will lead to a suspended page on Twitter. But some copies of the tweets as they originally appeared, including images, can be found by entering the links on web caches like archive.org and archive.is.

Acknowledgements If you publish using the data, please credit NBC News and include a link to this page. Send questions to <ben.popken@nbcuni.com>.

THIS Project

In this mark down, the russian troll data set and a sentiment data set that was pre 2014 was used for analysis. The Russian troll accounts were from September 2014 until September 2017. So I found tweets that were produced from April to May 2009. Therefore, camparisons of troll and non-troll data could be made.

``` r
library(ROCR)
```

    ## Warning: package 'ROCR' was built under R version 3.4.4

    ## Loading required package: gplots

    ## Warning: package 'gplots' was built under R version 3.4.4

    ## 
    ## Attaching package: 'gplots'

    ## The following object is masked from 'package:stats':
    ## 
    ##     lowess

``` r
library(caTools)# ROC, AUC
```

    ## Warning: package 'caTools' was built under R version 3.4.4

``` r
library(e1071) #  Naive Bayes 
```

    ## Warning: package 'e1071' was built under R version 3.4.4

``` r
library(SnowballC) # stemming
```

    ## Warning: package 'SnowballC' was built under R version 3.4.4

``` r
library(gmodels)
```

    ## Warning: package 'gmodels' was built under R version 3.4.4

``` r
library(tidytext)
```

    ## Warning: package 'tidytext' was built under R version 3.4.4

``` r
library(dplyr)
```

    ## Warning: package 'dplyr' was built under R version 3.4.4

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(ggplot2)
```

    ## Warning: package 'ggplot2' was built under R version 3.4.4

``` r
library(wordcloud)
```

    ## Warning: package 'wordcloud' was built under R version 3.4.4

    ## Loading required package: RColorBrewer

    ## 
    ## Attaching package: 'wordcloud'

    ## The following object is masked from 'package:gplots':
    ## 
    ##     textplot

``` r
library(tidyr)
```

    ## Warning: package 'tidyr' was built under R version 3.4.4

``` r
library(tm) # text data
```

    ## Warning: package 'tm' was built under R version 3.4.4

    ## Loading required package: NLP

    ## Warning: package 'NLP' was built under R version 3.4.4

    ## 
    ## Attaching package: 'NLP'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     annotate

``` r
library(effects) # regession models

#nt<-read.csv("C:/Users/John/Documents/R/russian_trolls/training.1600000.processed.noemoticon.csv")
#rt <-read.csv("file:///C:/Users/John/Documents/R/russian_trolls/tweets.csv/tweets.csv")

#set.seed(33)
#rt <- data.frame(sample_n(rt,6000,replace=FALSE))
#nt<-data.frame(sample_n(nt,6000,replace = FALSE))

#save(rt,file = "savedrt.RData")
#save(nt,file= "savednt.RData")
```

Because the CSV files were so large, two Rdata files were created. This helped with a shorter run time and committing to Github.

``` r
load("savednt.RData")

load("savedrt.RData")

#renameing Features
created_str <-as.Date(rt$created_str)
text<- as.character(rt$text)
colnames(nt)[6]<- as.character(c("text"))
colnames(nt)[3]<- "created_str"

#extracting columns
rtext<-select(rt,"text")
ntext<-select(nt,"text")


#adding a column Russian tweets
rtext$r_nr<-"r"

#adding column to non_russian tweets
ntext$r_nr<-"nr"
```

``` r
#combining data sets, sample function randomizes order of rows

tot_tweets<- rbind(ntext,rtext,stringAsfactors=FALSE)
tot_tweets <- tot_tweets[sample(nrow(tot_tweets)),]

#change character to factor
tot_tweets$r_nr<-factor(tot_tweets$r_nr, levels=c("r","nr"),ordered=TRUE)

#creating a corpus for the twitter text
text_corpus<- VCorpus(VectorSource(tot_tweets$text))
print(text_corpus)
```

    ## <<VCorpus>>
    ## Metadata:  corpus specific: 0, document level (indexed): 0
    ## Content:  documents: 12001

``` r
#cleaning tweets 
text_corpus_clean<-tm_map(text_corpus,content_transformer(tolower))

text_corpus_clean<-tm_map(text_corpus_clean,removeNumbers)

text_corpus_clean<-tm_map(text_corpus_clean,removeWords,stopwords())

text_corpus_clean<-tm_map(text_corpus_clean,removePunctuation)

                                                    text_corpus_clean<-tm_map(text_corpus_clean,stemDocument)
                                                                                                         
text_corpus_clean<-tm_map(text_corpus_clean,stripWhitespace)


#Tokenize the data
text_dtm <-DocumentTermMatrix(text_corpus_clean,control =
                                list(wordLengths=c(0,Inf)))
```

We separate the data into a training set and a test set. Then create the labels for the two sets.

``` r
text_dtm_train<- text_dtm[1:10000,]
text_dtm_test<- text_dtm[10001:12000,]


text_train_labels<- tot_tweets[1:10000,]$r_nr
text_test_labels<- tot_tweets[10001:12000,]$r_nr
```

We can now create three word clouds. The first wordcloud shows all of the data together. The second cloud show just the russian troll texts. The third is no Russian trolls at all.

``` r
#Overall word graph
wordcloud(text_corpus_clean, min.freq = 100,scale=c(2,.5),random.order = FALSE)
```

![](russian_trolls_files/figure-markdown_github/unnamed-chunk-5-1.png)

``` r
#Russian troll graph
rus<-subset(tot_tweets,r_nr=="r")
norus<-subset(tot_tweets,r_nr=="nr")

wordcloud(rus$text, max.words = 40,scale = c(3,.5))
```

    ## Warning in tm_map.SimpleCorpus(corpus, tm::removePunctuation):
    ## transformation drops documents

    ## Warning in tm_map.SimpleCorpus(corpus, function(x) tm::removeWords(x,
    ## tm::stopwords())): transformation drops documents

![](russian_trolls_files/figure-markdown_github/unnamed-chunk-5-2.png)

``` r
#no russian trolls graph
wordcloud(norus$text, max.words = 40, scale = c(3,.5))
```

    ## Warning in tm_map.SimpleCorpus(corpus, tm::removePunctuation):
    ## transformation drops documents

    ## Warning in tm_map.SimpleCorpus(corpus, tm::removePunctuation):
    ## transformation drops documents

![](russian_trolls_files/figure-markdown_github/unnamed-chunk-5-3.png)

``` r
#number of frequent terms. frenquency filter of words used less than 5 times
tweet_freq_words <- findFreqTerms(text_dtm_train, 5)
 
 str(tweet_freq_words)
```

    ##  chr [1:2611] "â" "â–¶" "â€¦" "â€˜" "â€“" "â€¢" "â€œ" "â«" "â»" "aâ€¦" ...

``` r
#create DTM
 
tweet_dtm_freq_train<- text_dtm_train[ , tweet_freq_words]
tweet_dtm_freq_test <- text_dtm_test[ , tweet_freq_words]

#Function for Change to categorical classifier, then apply to the columns
 convert_counts <- function(x) {
 x <- ifelse(x > 0, "Yes", "No")
 }

tweet_train <- apply(tweet_dtm_freq_train, MARGIN = 2,
 convert_counts)
tweet_test <- apply(tweet_dtm_freq_test, MARGIN = 2,
 convert_counts)
  
tweet_classifier<- naiveBayes(tweet_train,text_train_labels)
```

``` r
 tweet_test_pred <- predict(tweet_classifier, tweet_test)


 CrossTable(tweet_test_pred, text_test_labels,
 prop.chisq = FALSE, prop.t = FALSE,
 dnn = c('predicted', 'actual'))
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  2000 
    ## 
    ##  
    ##              | actual 
    ##    predicted |         r |        nr | Row Total | 
    ## -------------|-----------|-----------|-----------|
    ##            r |       890 |        21 |       911 | 
    ##              |     0.977 |     0.023 |     0.456 | 
    ##              |     0.900 |     0.021 |           | 
    ## -------------|-----------|-----------|-----------|
    ##           nr |        99 |       990 |      1089 | 
    ##              |     0.091 |     0.909 |     0.544 | 
    ##              |     0.100 |     0.979 |           | 
    ## -------------|-----------|-----------|-----------|
    ## Column Total |       989 |      1011 |      2000 | 
    ##              |     0.494 |     0.505 |           | 
    ## -------------|-----------|-----------|-----------|
    ## 
    ## 

``` r
#create ROC for Naive Bayes
troc<-predict(tweet_classifier,tweet_test, type = "raw")
predt<- prediction(troc[,"nr"],text_test_labels)

#calculate the area
auc<-performance(predt,"auc")


perf_r<- performance(predt, measure ='tpr',x.measure='fpr')
plot(perf_r,colorize=T)
```

![](russian_trolls_files/figure-markdown_github/unnamed-chunk-8-1.png)

``` r
    # abline(a=0,b=1))
     #legend(.6,.2,auc, title = "AUC"),
     #main= "ROC Curve")
print(auc)
```

    ## An object of class "performance"
    ## Slot "x.name":
    ## [1] "None"
    ## 
    ## Slot "y.name":
    ## [1] "Area under the ROC curve"
    ## 
    ## Slot "alpha.name":
    ## [1] "none"
    ## 
    ## Slot "x.values":
    ## list()
    ## 
    ## Slot "y.values":
    ## [[1]]
    ## [1] 0.983643
    ## 
    ## 
    ## Slot "alpha.values":
    ## list()

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.4.4

    ## Loading required package: lattice

``` r
 #logistic Regression

sparse_dtm<-removeSparseTerms(text_dtm, 0.99) #terms appear in more than 1% of tweets

#new Data frame
tweetsSparse<-as.data.frame(as.matrix(sparse_dtm))
colnames(tweetsSparse)<-make.names(colnames(tweetsSparse))
tweetsSparse$r_nr<-tot_tweets$r_nr

#split the data set
set.seed(200)

split<- sample.split(tweetsSparse,SplitRatio=2/3)

trainSparse<- subset(tweetsSparse, split=="TRUE")#in split 
testSparse<- subset(tweetsSparse, split=="FALSE")#not in split

tweet.logit <- glm(r_nr ~ ., data = trainSparse, family = "binomial")
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
tweet.logit.test<-predict(tweet.logit,type = "response", newdata = testSparse)


cmatrix_logregr<-table(testSparse$r_nr, tweet.logit.test>0.5)

cmatrix_logregr
```

    ##     
    ##      FALSE TRUE
    ##   r   1695  332
    ##   nr    22 2037

``` r
tweet.logit.test1<-predict(tweet.logit, type = "response", newdata = trainSparse)
cmatrix1 <-table(trainSparse$r_nr,tweet.logit.test1>0.5)

cmatrix1
```

    ##     
    ##      FALSE TRUE
    ##   r   3268  705
    ##   nr    38 3903

``` r
#Descision tree
library(rpart)
```

    ## Warning: package 'rpart' was built under R version 3.4.4

``` r
library(rpart.plot)
```

    ## Warning: package 'rpart.plot' was built under R version 3.4.4

``` r
tweetCART <- rpart(r_nr ~ . , data = trainSparse, method = "class")
prp(tweetCART)
```

![](russian_trolls_files/figure-markdown_github/unnamed-chunk-10-1.png) If any of the three terms rt,trump, or clinton showed themselves in the Russian troll accounts.

Overall Naive Bayes return the most promising results for filtering tweets against a wide range of tweets.
