R Notebook
================

This is an [R Markdown](http://rmarkdown.rstudio.com) Notebook. When you execute code within the notebook, the results appear beneath the code.

Context As part of the House Intelligence Committee investigation into how Russia may have influenced the 2016 US Election, Twitter released the screen names of almost 3000 Twitter accounts believed to be connected to Russia’s Internet Research Agency, a company known for operating social media troll accounts. Twitter immediately suspended these accounts, deleting their data from Twitter.com and the Twitter API. A team at NBC News including Ben Popken and EJ Fox was able to reconstruct a dataset consisting of a subset of the deleted data for their investigation and were able to show how these troll accounts went on attack during key election moments. This dataset is the body of this open-sourced reconstruction.

For more background, read the NBC news article publicizing the release: "Twitter deleted 200,000 Russian troll tweets. Read them here."

Content This dataset contains two CSV files. tweets.csv includes details on individual tweets, while users.csv includes details on individual accounts.

To recreate a link to an individual tweet found in the dataset, replace user\_key in <https://twitter.com/user_key/status/tweet_id> with the screen-name from the user\_key field and tweet\_id with the number in the tweet\_id field.

Following the links will lead to a suspended page on Twitter. But some copies of the tweets as they originally appeared, including images, can be found by entering the links on web caches like archive.org and archive.is.

Acknowledgements If you publish using the data, please credit NBC News and include a link to this page. Send questions to <ben.popken@nbcuni.com>.

``` r
library(rtweet)
```

    ## Warning: package 'rtweet' was built under R version 3.4.4

``` r
library(twitteR)
```

    ## Warning: package 'twitteR' was built under R version 3.4.4

    ## 
    ## Attaching package: 'twitteR'

    ## The following object is masked from 'package:rtweet':
    ## 
    ##     lookup_statuses

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

    ## The following objects are masked from 'package:twitteR':
    ## 
    ##     id, location

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

``` r
library(tidyr)
```

    ## Warning: package 'tidyr' was built under R version 3.4.4

``` r
library(tm)
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
rt <-read.csv("file:///C:/Users/John/Documents/R/russian_trolls/tweets.csv/tweets.csv")

user.rt<-read.csv("C:/Users/John/Documents/R/russian_trolls/users.csv")

#renameing Features
created_str <-as.Date(rt$created_str)

str(rt)
```

    ## 'data.frame':    203482 obs. of  16 variables:
    ##  $ user_id              : num  1.87e+09 2.57e+09 1.71e+09 2.58e+09 1.77e+09 ...
    ##  $ user_key             : Factor w/ 454 levels "_billy_moyer_",..: 374 117 97 348 294 151 38 224 22 168 ...
    ##  $ created_at           : num  1.46e+12 1.48e+12 1.49e+12 1.48e+12 1.50e+12 ...
    ##  $ created_str          : Factor w/ 198422 levels "","2014-07-14 18:04:55",..: 32435 91101 176773 145243 197791 104222 181292 147779 106367 31601 ...
    ##  $ retweet_count        : int  NA 0 NA NA NA NA NA NA 0 NA ...
    ##  $ retweeted            : Factor w/ 2 levels "","false": 1 2 1 1 1 1 1 1 2 1 ...
    ##  $ favorite_count       : int  NA 0 NA NA NA NA NA NA 0 NA ...
    ##  $ text                 : Factor w/ 174986 levels "","'#SickHillary refuses to answer question about concussion, walks away. #LaueringTheBar\nhttps://t.co/7DK8P8yiC0"| __truncated__,..: 5381 21469 109742 101811 139848 14826 112855 42146 49739 39072 ...
    ##  $ tweet_id             : num  7.12e+17 7.86e+17 8.34e+17 8.13e+17 8.94e+17 ...
    ##  $ source               : Factor w/ 20 levels "","<a href=\"http://bufferapp.com\" rel=\"nofollow\">Buffer</a>",..: 1 15 1 1 1 1 1 1 11 1 ...
    ##  $ hashtags             : Factor w/ 18343 levels "[\"_Malikalovess\"]",..: 7768 18343 18343 2484 18343 18343 18343 18343 18343 1626 ...
    ##  $ expanded_urls        : Factor w/ 22215 levels "[\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\"]",..: 22215 5456 22215 22215 22215 22215 22215 22215 11374 22215 ...
    ##  $ posted               : Factor w/ 1 level "POSTED": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ mentions             : Factor w/ 16683 levels "[\"___lorraine__\"]",..: 16683 16683 16683 16683 16683 16683 16683 16683 881 16683 ...
    ##  $ retweeted_status_id  : num  NA NA NA NA NA ...
    ##  $ in_reply_to_status_id: num  NA NA NA NA NA NA NA NA NA NA ...

``` r
summary(rt)
```

    ##     user_id                    user_key        created_at       
    ##  Min.   :1.871e+07   ameliebaldwin :  9269   Min.   :1.405e+12  
    ##  1st Qu.:1.671e+09   hyddrox       :  6813   1st Qu.:1.471e+12  
    ##  Median :1.857e+09   giselleevns   :  6652   Median :1.477e+12  
    ##  Mean   :1.404e+16   patriotblake  :  4140   Mean   :1.473e+12  
    ##  3rd Qu.:2.590e+09   thefoundingson:  3663   3rd Qu.:1.483e+12  
    ##  Max.   :7.893e+17   melvinsroberts:  3346   Max.   :1.506e+12  
    ##  NA's   :8065        (Other)       :169599   NA's   :21         
    ##               created_str     retweet_count      retweeted     
    ##                     :    21   Min.   :    0.00        :145399  
    ##  2016-02-03 12:42:11:     6   1st Qu.:    0.00   false: 58083  
    ##  2016-02-05 12:08:51:     6   Median :    0.00                 
    ##  2016-02-11 07:39:05:     6   Mean   :   39.64                 
    ##  2016-02-11 07:59:33:     6   3rd Qu.:    0.00                 
    ##  2016-02-14 18:59:29:     6   Max.   :20494.00                 
    ##  (Other)            :203431   NA's   :145399                   
    ##  favorite_count   
    ##  Min.   :    0.0  
    ##  1st Qu.:    0.0  
    ##  Median :    0.0  
    ##  Mean   :   35.5  
    ##  3rd Qu.:    0.0  
    ##  Max.   :26655.0  
    ##  NA's   :145399   
    ##                                                                                                                                  text       
    ##  RT @MarkAlmost: MT @jstines3: Dear LORD, please bless                                                                             :    30  
    ##                                                                                                                                    :    21  
    ##  RT @AtomicElbow1: Trump twitter suspended #2016ElectionIn3Words                                                                   :    17  
    ##  RT @lgmaterna: Anyone voting Clinton #LostIn3Words                                                                                :    16  
    ##  RT @The_Anti_Fox: 7yrs                                                                                                            :    16  
    ##  RT @AndyHashtagger: #TrumpsFavoriteHeadline An army of Trump clones is ready to fight and serve its master https://t.co/HQgXvUtVsQ:    15  
    ##  (Other)                                                                                                                           :203367  
    ##     tweet_id        
    ##  Min.   :4.887e+17  
    ##  1st Qu.:7.655e+17  
    ##  Median :7.888e+17  
    ##  Mean   :7.735e+17  
    ##  3rd Qu.:8.153e+17  
    ##  Max.   :9.126e+17  
    ##  NA's   :2314       
    ##                                                                                  source      
    ##                                                                                     :145398  
    ##  <a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>                 : 42685  
    ##  <a href="http://twitterfeed.com" rel="nofollow">twitterfeed</a>                    :  6926  
    ##  <a href="https://about.twitter.com/products/tweetdeck" rel="nofollow">TweetDeck</a>:  6410  
    ##  <a href="http://twibble.io" rel="nofollow">Twibble.io</a>                          :  1491  
    ##  <a href="http://dlvr.it" rel="nofollow">dlvr.it</a>                                :   243  
    ##  (Other)                                                                            :   329  
    ##                      hashtags                  expanded_urls   
    ##  []                      :114696   []                 :173789  
    ##  ["Politics"]            :  3143   [""]               :  2538  
    ##  ["news"]                :  1469   ["",""]            :   854  
    ##  ["tcot"]                :  1033   ["","",""]         :   338  
    ##  ["MerkelMussBleiben"]   :   796   ["","","",""]      :   202  
    ##  ["RejectedDebateTopics"]:   614   ["","","","","",""]:   101  
    ##  (Other)                 : 81731   (Other)            : 25660  
    ##     posted                      mentions      retweeted_status_id
    ##  POSTED:203482   []                 :163964   Min.   :7.676e+16  
    ##                  ["realdonaldtrump"]:   658   1st Qu.:7.769e+17  
    ##                  ["hillaryclinton"] :   315   Median :7.838e+17  
    ##                  ["lindasuhler"]    :   245   Mean   :7.809e+17  
    ##                  ["ten_gop"]        :   205   3rd Qu.:7.893e+17  
    ##                  ["petefrt"]        :   184   Max.   :8.927e+17  
    ##                  (Other)            : 37911   NA's   :163831     
    ##  in_reply_to_status_id
    ##  Min.   :6.108e+17    
    ##  1st Qu.:7.627e+17    
    ##  Median :7.736e+17    
    ##  Mean   :7.719e+17    
    ##  3rd Qu.:7.814e+17    
    ##  Max.   :8.010e+17    
    ##  NA's   :202923

``` r
text_corpus<- VCorpus(VectorSource(rt$text))
print(text_corpus)
```

    ## <<VCorpus>>
    ## Metadata:  corpus specific: 0, document level (indexed): 0
    ## Content:  documents: 203482

``` r
text_dtm <- DocumentTermMatrix(text_corpus, control = list(
 tolower = TRUE,
 removeNumbers= TRUE,
 stopwords=TRUE,
removePunctuation= TRUE,
stemming =TRUE )) 
```

``` r
text_dtm_train<- text_dtm[1:152600,]
text_dtm_test<- text_dtm[152601:203482,]
```

Add a new chunk by clicking the *Insert Chunk* button on the toolbar or by pressing *Ctrl+Alt+I*.

When you save the notebook, an HTML file containing the code and output will be saved alongside it (click the *Preview* button or press *Ctrl+Shift+K* to preview the HTML file).

The preview shows you a rendered HTML copy of the contents of the editor. Consequently, unlike *Knit*, *Preview* does not run any R code chunks. Instead, the output of the chunk when it was last run in the editor is displayed.
