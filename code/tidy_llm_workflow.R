devtools::install_github("AlbertRapp/tidychatmodels")

readRenviron(".Renviron")

library(tidychatmodels)

# select vendor and enter access key
create_chat('openai', Sys.getenv('MY_OPEN_AI_API_KEY')) |>
  add_model('gpt-4o-mini') |>
  # adjust temperature ("creativity") and output length
  add_params(temperature = 0.5, max_tokens = 100) |>
  # enter system-level prompt
  add_message(
    role = 'system',
    message = paste('You are a chatbot that classifies',
                    'open-ended responses in public',
                    'opinion surveys.')
  ) |> 
  # enter user-level prompt
  add_message(
    message = paste('Classify the following open-ended response',
                    'to the standard most important issue question',
                    'into a political issue topic:',
                    '"Immigration reform is essential for a fair, legal system."')
  ) |> 
  # send to LLM
  perform_chat() |> 
  extract_chat()