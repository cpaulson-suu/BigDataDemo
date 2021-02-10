library(RTextTools)
library(maxent) #low-memory multinomial logistic regression, short for "maximum entropy"
library(SnowballC)
library(ggplot2)

tweets <- read.csv("tweets_subset.csv")
# This has two features (class and text) and 20 observations 
# Randomly shuffled. Training: Rows 1:14 Test 15:20

tweets$text <- as.character(tweets$text)

#build tdm (term document matrix)
#note: removeSparseTerms means, remove words occuring in fewer than 5% of the documents
#the other parameters are more obvious to interpret
#reminder: type ?create_matrix to see other options, for instance weighting=tm:weightTfIdf
matrix <- create_matrix(tweets$text, language="english", removeSparseTerms = 0.95, removeStopwords=TRUE, removeNumbers=TRUE, stemWords=TRUE, stripWhitespace=TRUE, toLower=TRUE)

mat <- as.matrix(matrix)
model <- maxent(mat[1:14], tweets[1:14,1])
Predictions <- predict(model, mat[15:20,])
table(tweets[15:20,1],as.factor(Predictions[,1]),dnn=c("Actual", "Predicted"))
recall_accuracy(tweets[15:20,1], Predictions)


##can also use the TM package
# we assume input in the form of a vector where each component is a text document.
# the VectorSource command takes a vector X of texts and interprets each element as a document
# the SimpleCorpus command is used to create a corpus

#separate out training and test
traindata <- tweets[1:14,]
testdata <- tweets[15:20,]

#create a vector of texts
trainvector <- as.vector(traindata$text)
testvector <- as.vector(testdata$text)

#create source for vectors (interpret each element as a document)
trainsource <- VectorSource(trainvector)
testsource <- VectorSource(testvector)

#create corpus for data
traincorpus <- SimpleCorpus(trainsource)
testcorpus <- SimpleCorpus(testsource)

#do transformations on the data
#these are on the slides from class
traincorpus <- tm_map(traincorpus, content_transformer(stripWhitespace))
traincorpus <- tm_map(traincorpus,content_transformer(tolower))
traincorpus <- tm_map(traincorpus,content_transformer(removeWords),stopwords("english"))
traincorpus <- tm_map(traincorpus,content_transformer(removePunctuation))
traincorpus <- tm_map(traincorpus,content_transformer(removeNumbers))

testcorpus <- tm_map(testcorpus,content_transformer(stripWhitespace))
testcorpus <- tm_map(testcorpus,content_transformer(tolower))
testcorpus <- tm_map(testcorpus,content_transformer(removeWords),stopwords("english"))
testcorpus <- tm_map(testcorpus,content_transformer(removePunctuation))

#stemming

#keep a copy of original corpus to use later as a dictionary for stem completion
CorpusCopy <- traincorpus

#stem words
traincorpus <- tm_map(traincorpus, content_transformer(stemDocument))

#unexpectedly, stemCompletion completes an empty string to a word in the dictionary
#remove empty string to avoid
#note: I can't vouch for this code!
stemCompletion2 <- function(x, dictionary){
  x <- unlist(strsplit(as.character(x), " "))
  x <- x[x != ""]
  x <- stemCompletion(x, dictionary=dictionary)
  x <- paste(x, sep="",collapse=" ")
  PlainTextDocument(stripWhitespace(x))
}
traincorpus <- lapply(traincorpus, stemCompletion2, dictionary=CorpusCopy)
temp <- as.matrix(lapply(traincorpus, '[[',1))
temp <- data.frame(temp)
traincorpus <- Corpus(VectorSource(temp$temp))


#create a term/document matrix
tdm1 <- TermDocumentMatrix(traincorpus)
#remove rare words
tdm1 <- removeSparseTerms(tdm1, .95)
#we need to transpose these matrices
trainmatrix <- t(tdm1);
testmatrix <- t(TermDocumentMatrix(testcorpus));

#train logistic regression model
#of course, we could try other procedures/models as well
model <- maxent(as.matrix(trainmatrix), as.factor(traindata$class))
Predictions <- predict(model,as.matrix(testmatrix))
table(testdata$class, Predictions[,1], dnn=c("Actual", "Predicted"))


##exploratory tools

#once the TDM is computed, we can do some exploratory analysis
freq.terms <- findFreqTerms(tdm1, lowfreq = 2)

term.freq <- rowSums(as.matrix(tdm1))
term.freq <- subset(term.freq, term.freq>=2)
df2 <- data.frame(term=names(term.freq), freq = term.freq)
ggplot(df2, aes(x=term,y=freq)) + geom_bar(stat="identity") + xlab("Terms")+ylab("Count") + coord_flip()


##correlations: which words are associated with "apple"?
findAssocs(tdm1, "apple", 0.05)

#which words are associated with 'ios'?
findAssocs(tdm1, "ios", 0.05)
