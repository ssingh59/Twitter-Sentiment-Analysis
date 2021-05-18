
#****************************************************************************************************************************************************
## Project1- Twitter Sentiment analysis for Covid-19 in India using TM and other modules##
## Group Name : Group 12
## Group Member Name : 
##    Manasa Geemarahalli Prakash 10457598
##    Mahvish Syed 10456845
##    Shweta Singh 10457493
##    Glen D’mello 10452455
##
##
#****************************************************************************************************************************************************


rm(list= ls()) # to clear the global enviornment

#Installation of the needed packages
#****************************************************************************************************************************************************
list.of.packages <- c("twitteR", "tm", "SnowballC", "ggplot2", "RColorBrewer", "wordcloud", "topicmodels", "data.table" ,"syuzhet","lubridate","scales","reshape2","qplyr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
install.packages("qplyr")
install.packages("rtweet")
install.packages("twitteR")
library("twitteR")
library(rtweet)
library(dplyr)
#****************************************************************************************************************************************************


#****************************************************************************************************************************************************
## Extraction Of the Data From Twitter  
#****************************************************************************************************************************************************    

setup_twitter_oauth('pq9cNg3BUU2EqIUDbi75oRRdq', 'P88EPX1l1aMnoDnFtdsO3iOZcFveKM5vIuLUNuo3W1aDzichFd', '118425371-Bnx7lyzFWytRzORMCKtgHMTx226wpbtgC7QqpcOF',
                    'f07cgIaaCkpxT4B64EJBqQ560iNIZiGRf1kisb3mClceW')

covidTweets <- search_tweets("#covidIndia", n = 4000, include_rts = FALSE, lang="en")

n.tweet <- length(covidTweets)

tweets.df <- as.data.frame(covidTweets) #converting the lists of tweets into dataframe 
View(tweets.df)
new_df <- tweets.df[,5]  ## Selecting only the 5th column from the data frame 
View(new_df)


#write.csv(new_df, file = "/Users/mahvishsyed/Desktop/Covid.csv") # saving this dataframe into a csv_file for offline use or incase 

#****************************************************************************************************************************************************
## Transformation and Cleaning of the Data- Removing stop words, urls , numeric data and other stuff
#****************************************************************************************************************************************************

library(tm)

View(new_df)
# build a corpus, and specify the source to be character vectors
myCorpus <-Corpus(VectorSource(iconv(new_df, "utf-8", "ASCII", sub="")))

x<-as.character(myCorpus) # We will need this for sentiment analysis so just copying the corpus to a varaible x
inspect(myCorpus[1:6]) # inspect the 1st 6 lines form the corpus

# convert to lower case
myCorpus <- tm_map(myCorpus, content_transformer(tolower))

# remove URLs
removeURL <- function(x) gsub("http[^[:space:]]*","", x)
myCorpus <-tm_map(myCorpus, content_transformer(removeURL))
#inspect(myCorpus[1:6])

# remove anything other than English letters or space
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*","",x)
myCorpus <- tm_map(myCorpus, content_transformer(removeNumPunct))
#inspect(myCorpus[1:6])


# Removing extra add words which I feel are unnessary
myStopwords <- c(stopwords('english'),"This","a","of","the","was","is", "that", "covid", "covidindia", "india")


myCorpus <- tm_map(myCorpus, removeWords, myStopwords)

#remove extra whitespace
myCorpus <- tm_map(myCorpus, stripWhitespace)
# keep a copy of corpus to use later as a dictionary for stem completion
myCorpus <- tm_map(myCorpus, removeNumbers)
inspect(myCorpus[1:10])
myCorpusCopy <- myCorpus


#****************************************************************************************************************************************************
## Loading Data 
#****************************************************************************************************************************************************
tdm <- TermDocumentMatrix(myCorpus, control = list(wordLengths = c(1,Inf)))
?TermDocumentMatrix
tdm
inspect(tdm)


#****************************************************************************************************************************************************
## Frequency Plotting  
#****************************************************************************************************************************************************
#inspect frequent words
(freq.terms <- findFreqTerms(tdm, lowfreq = 100)) # finding word freq here i am only considering words with occcur in atleast 100 docs to
# avoid over crowding 

# Creating a named vector with the frequency of the words
term.freq <- rowSums(as.matrix(tdm))
term.freq <- subset(term.freq, term.freq >= 100)
# Transforming a named vector to a dataframe
df <- data.frame(term = names(term.freq), freq = term.freq)

library(ggplot2)
p <- ggplot(df, aes(x = term, y = freq)) 
p <- p +  geom_bar(stat = "identity") 
p <- p +  xlab("Terms") + ylab("Count")
p <- p + coord_flip()
print(p)  #ploting the freq words on histograms


#****************************************************************************************************************************************************
# Word Association and Corelation 
#****************************************************************************************************************************************************

(freq.terms <- findFreqTerms(tdm, lowfreq = 50))

# Finding some associations
findAssocs(tdm, "covid", 0.1) # findAssocs is used to find the association of various words with 
#respect to a particular word provided in " ".
findAssocs(tdm, "vaccine", 0.2)
findAssocs(tdm, "government", 0.1)


#****************************************************************************************************************************************************
#Sentiment Analysis
#*********************************************************************************************************************
library(syuzhet)
library(lubridate)
library( ggplot2)
library(reshape2)
library(qplyr)

sentiment <- get_nrc_sentiment(x)


t<-as.matrix(sentiment)
#write.csv(sentiment, ""/Users/mahvishsyed/Desktop/CS-553/Text Mining Project/MySentiments.csv")

getwd()

comments <- cbind(new_df,sentiment)

sentimentTotals <- data.frame(colSums(sentiment[,c(1:8)]))
names(sentimentTotals) <- "count"
sentimentTotals <- cbind("sentiment" = rownames(sentimentTotals), sentimentTotals)
rownames(sentimentTotals) <- NULL

ggplot(data = sentimentTotals, aes(x = sentiment, y = count)) +
  geom_bar(aes(fill = sentiment), stat = "identity") +
  theme(legend.position = "none") +
  xlab("Sentiment") + ylab("Total Count") + ggtitle("Total Sentiment Score for all Tweets")



#****************************************************************************************************************************************************
## Hiearchial clustering
#*******************************************************************************************************************************************************
# remove sparse terms
tdm2 <- removeSparseTerms(tdm, sparse = 0.9)
# Showing the terms that are left for the analysis
print(dimnames(tdm2)$Terms)

m2 <- as.matrix(tdm2)

# cluster terms
distMatrix <-dist(scale(m2))
fit <- hclust(distMatrix, method = "ward.D2")

p <- plot(fit)
p <- rect.hclust(fit, k = 4) # fit into 6 clusters
print(p)
# Showing the groups
(groups <-cutree(fit, k = 4))
print(groups)

#********************************************************************************************
#Topic Modelling
#********************************************************************************************
dtm <- as.DocumentTermMatrix(tdm)
library(topicmodels)
library(data.table)

lda <- LDA(dtm, k = 8) # find 8 topics
(term <- terms(lda,6)) # first 6 terms of every topic

term <- apply(term, MARGIN = 2, paste, collapse = ", ")

# first topic identified for every document (tweet)
topic <- topics(lda, 1)
topics <- data.frame(date=as.IDate(tweets.df$created), topic)
p <- qplot(date, ..count.., data=topics, geom = "density", fill = term[topic], position="stack")
print(p)


