library(tidyverse)  # tidy data manipulation
library(tidytext)   # tidy NLP
library(SnowballC)  # word stemming  

# 1. Create/prepare a corpus ----------------------------------------------

# Fake polling data with responses to the question
polling_data <- data.frame(
  id = 1:5,
  response = c(
    "The economy is slowing. Inflation harms family budgets greatly.",
    "Immigration reform is essential for a fair, legal system.",
    "Economic growth and environmental pollution are top issues.",
    "Education reform is critical for students' future success.",
    "Immigrants, both legal and illegal, need government support."
  )
)
# Convert to tidy format
tidy_polling <- polling_data %>%
  unnest_tokens(word, response)

# 2. Clean text -----------------------------------------------------------

# Remove stop words
data("stop_words")
cleaned_polling <- tidy_polling %>%
  anti_join(stop_words)

# Apply stemming
stemmed_polling <- cleaned_polling %>%
  mutate(word = wordStem(word))

# View stemmed data
stemmed_polling %>%
  count(word) %>%
  arrange(desc(n)) %>%
  head()

# 3. Sentiment Analysis  --------------------------------------------------

# Perform sentiment analysis
sentiment_polling <- cleaned_polling %>%
  inner_join(get_sentiments("bing")) %>%
  count(sentiment)

# View sentiment results
sentiment_polling

# 4. Analyzing TF-IDF -----------------------------------------------------

# Add document-term frequencies
word_counts <- cleaned_polling %>%
  count(id, word, sort = TRUE)

# Calculate tf-idf
polling_tf_idf <- word_counts %>%
  bind_tf_idf(word, id, n)

# View tf-idf results
polling_tf_idf %>%
  arrange(desc(tf_idf)) %>%
  head(10)

# 5. Topic Modeling -------------------------------------------------------
library(topicmodels)
library(tm)

# Prepare document-term matrix
dtm <- DocumentTermMatrix(Corpus(VectorSource(polling_data$response)))

# Fit LDA model with 2 topics
lda_model <- LDA(dtm, k = 3, control = list(seed = 123))

# View top terms per topic
terms(lda_model, 5)



