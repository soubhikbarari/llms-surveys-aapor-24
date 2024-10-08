
# [Total Survey Error](https://en.wikipedia.org/wiki/Total_survey_error)
The difference between a population parameter (such as the mean, total or proportion) and the estimate of that parameter based on the sample survey or census, which can be decomposed into the following sources of error.

- *Measurement-Related Errors:*
    - *Construct:*
        - **Validity Error**: Occurs when the survey does not measure what it intends to measure.
    - *Measurement:*
        - **Measurement Error**: The difference between the actual value and the value obtained in the survey.
    - *Response:*
        - **Processing Error**: Mistakes in data entry, coding, or other post-collection processing.

- *Representation-Related Errors:*
    - *Target Population:*
        - **Coverage Error**: Error that occurs when some members of the target population are not included in the survey frame.
    - *Sampling Frame:*
        - **Sampling Error**: The error caused by observing a sample instead of the whole population.
    - *Sample:*
        - **Nonresponse Error**: The bias that results when certain groups of respondents are systematically less likely to respond.
    - *Respondents:*
        - **Adjustment Error**: Error introduced during post-survey adjustments, such as weighting.

# Natural Language Processing (NLP)
Computational tools to understand human language.

- *NLP Structure:*
    - **Token**: The smallest unit in text, such as a word or punctuation mark.
        - **N-Gram**: A sequence of 'n' items from a given sample of text.
    - **Document**: A sequence of tokens at a meaningful level (i.e., a question response) for a particular analysis.
    - **Stop Words**: Commonly used words (like 'and', 'the') that are often filtered out in text analysis.
    - **Document-Term Matrix**: A matrix that describes the frequency of terms that occur in a collection of documents.
    - **Corpus**: A large collection of similarly-themed documents used for a particular analysis.

- *NLP Representation:*
    - **Bag-of-Words**: A model that represents text as a collection of words, disregarding grammar and word order.
    - **Dictionary**: A grouping of semantically similar or identical individual tokens into common categories (e.g., sentiment like positive, negative; part of speech like verbs, nouns).
    - **Embedding**: A numerical representation of words in a continuous vector space.

- *NLP Tasks:*
    - **TF-IDF (Term Frequency-Inverse Document Frequency)**: A weighting scheme that reflects the importance of a word in a document relative to a collection of documents.
    - **Supervised Classification**: An technique in machine learning where the goal is to categorize (classify) text data into predefined categories (classes). This is done using labeled training data.
        - **Sentiment Analysis**: A type of supervised learning method that datermines the emotional tone (positive, negative, or neutral) behind a body of text. It's commonly used for analyzing opinions in social media or customer reviews.
    - **Unsupervised Learning**: A technique in machine learning to find patterns or structure in textual data (without any labels, metadata, or external information). In NLP, this is used for tasks like clustering or topic modeling where labels aren't predefined.
        - **Topic Modelling**: A type of unsupervised learning method that discovers abstract topics within a collection of documents. It uncovers hidden structures or themes without needing labeled data. Common algorithms for topic modelling include Latent Dirichlet Allocation (LDA).
    - **Named Entity Recognition (NER)**: A process in which the system identifies and classifies named entities (e.g., people, organizations, locations) mentioned in a body of text. NER is key in extracting important information from unstructured text.
    - **Part of Speech (POS) Tagging**: The task of labeling each word in a sentence with its corresponding part of speech (noun, verb, adjective, etc.). POS tagging helps in understanding the grammatical structure of a sentence.
    - **Stance Detection**: The task of extracting and identifying the position (for, against, neutral) expressed by a speaker or author or other entity in relation to a given topic. It differs from sentiment analysis in that it concerns a particular entity and a target.
    - **Text Summarization**: The task of automatically generating a concise and coherent version of a longer text document while preserving key information.
    - **Text Generation**: The process of generating coherent and contextually relevant text given a starting prompt.

- *NLP Preprocessing:*
    - **Stemming**: The process of reducing words to their base form (e.g. "running" becomes "run").
    - **Lemmatization**: A more advanced form of stemming where words are reduced to their base or dictionary form (lemma), considering the context (e.g., "better" would be reduced to "good" rather than "bet")
    - **Stop Word Removal**: The process of removing common words like “and,” “the,” or “is” that carry less semantic value for an NLP task.

- *NLP Packages:*
    - [`quanteda`](https://quanteda.io/): A package for quantitative text analysis in R, known for its fast and flexible implementation of various NLP tasks such as tokenization, word frequency analysis, and topic modeling.
    - [`tidytext`](https://cran.r-project.org/web/packages/tidytext/vignettes/tidytext.html): An R package that applies principles of ["tidy data"](https://r4ds.had.co.nz/tidy-data.html) to text processing, making it easier to manipulate text data using tools like [`dplyr`](https://dplyr.tidyverse.org/) and [`ggplot2`](https://ggplot2.tidyverse.org/).
    - [`text2vec`](https://text2vec.org/): A powerful R package for text vectorization and topic modeling. It supports techniques like word embeddings, LDA, and other NLP tasks, offering a fast and memory-efficient implementation.
    - [`nltk`](https://www.nltk.org/): The Natural Language Toolkit is a widely used Python library for various NLP tasks, such as tokenization, POS tagging, parsing, and more. It is a comprehensive resource for both beginners and researchers.
    - [`vader`](https://aaai.org/ocs/ICWSM/ICWSM14/paper/view/8109/8122): A method specifically tailored for sentiment analysis (implemented in both [R](https://cran.r-project.org/web/packages/vader/index.html) and [Python](https://github.com/cjhutto/vaderSentiment)) particularly for social media applications. It uses a rule-based method to analyze the sentiment of a text (positive, neutral, or negative).
    - `gensim`: A robust Python library for topic modeling and document similarity analysis. It is well-known for its efficient implementation of LDA and [Word2vec](https://en.wikipedia.org/wiki/Word2vec) models.
    - [`stm`](https://www.structuraltopicmodel.com/): Short for [Structural Topic Model](https://projects.iq.harvard.edu/files/wcfia/files/stmnips2013.pdf), this R package is designed for estimating topic models with document-level covariates. It helps in understanding how the topics in a document change over time or by other metadata.
    - [`spaCy`](https://spacy.io/): A popular and fast NLP library in Python, used for tasks such as tokenization, POS tagging, named entity recognition, and dependency parsing. It is designed for production use and supports large-scale text processing.

# Generative AI
Machine learning (ML) methods to understand and generate high-dimensional data (text, audio, video).

- **Large Language Models (LLM)**: Models drawing on techniques from both machine learning and natural language processing that can generate human-like text based on vast amounts of data.
    
- *LLM Structure:*
    - **Input / Prompt**: The initial text or instruction given to a model to generate a response (e.g., asking a model to summarize an article or answer a question).
        - **System Prompt**:  A prompt given to the model to control its overall behavior across subsequent prompts. System prompts are typically not exposed to the user in LLM-based applications (e.g., ChatGPT) unless the user is working with a playground / sandbox environment or directly building / calling an LLM programatically.
        - **User Prompt**: The text entered by the user to request information or a task from the model. It can be a question, statement, or directive, such as "Explain the importance of attention in transformers."
    - **Output / Response / Completion**: The text generated by the model in response to the input prompt (e.g., the summary generated for a particular article based on a prompt to do so).
    - **Neural Network**: A machine learning model inspired by the structure of the human brain. It consists of interconnected layers of nodes (neurons) that process input data and learn patterns through training.
    - **Transformer**: A type of neural network architecture used in NLP models. It was introduced in the landmark paper ["Attention is All You Need"](https://arxiv.org/pdf/1706.03762) and is the backbone of many modern NLP models, including GPT. Unlike traditional [RNNs (Recurrent Neural Networks)](https://en.wikipedia.org/wiki/Recurrent_neural_network), its immediate predecessor, the transformer model can process input data in parallel, leading to greater efficiency, especially for large datasets.
    - **Attention**: A mechanism in neural networks that allows the model to focus on specific parts of the input sequence when making predictions. It assigns different weights to different words or tokens based on their relevance to the task at hand. For example, in a translation task, the attention mechanism helps the model focus on the relevant words in the source sentence that correspond to the words being generated in the target sentence.
        - **Multi-Head Attention**: A specific type of attention mechanism used in transformers, where the model runs multiple attention layers (heads) in parallel. Each head focuses on different parts of the input sequence, and their results are combined to improve the model’s ability to understand complex relationships in the data. This approach allows the model to capture multiple levels of meaning or dependencies within the input text.
    - **Encoder**: In a transformer model, the encoder is responsible for processing the input data (e.g., a sentence in a translation task). It converts the input text into a set of continuous representations, capturing the relationships between different words or tokens. The encoder is often used in tasks like translation or summarization, where it creates a rich understanding of the input sequence.
    - **Decoder**: The decoder is the counterpart to the encoder and is responsible for generating the output sequence (e.g., the translated sentence or a summary). In models like GPT, the decoder takes the encoded information from the input and generates text step by step, word by word. It plays a crucial role in autoregressive tasks, where the output is generated in sequence based on previous tokens.

- *LLM Preparation:*
    - **Pre-Training**: Initial training of the model on a large dataset (very expensive, time-consuming).
    - **Fine-Tuning**: Adjusting a pre-trained model on a smaller, specific dataset for a particular task or context (e.g. survey question or survey project).

- *LLM Utilization:*
    - **Prompt Engineering**
        - *Prompt Methods:*
            - **Zero-Shot Prompting**: No examples are provided, and the model is expected to answer based on general knowledge.
            - **Few-Shot Prompting**: A few examples are provided to guide the model.
            - **Many-Shot Prompting**: Several examples are provided to demonstrate the task.
            - **Chain of Thought Prompting**: Encourages the model to break down its reasoning step by step.
        - *Prompt Concepts:*
            - **Objective**: The specific goal or task the prompt is designed to achieve (e.g., generating text, summarizing content).
            - **Persona**: The role or character that the LLM is instructed to adopt when responding (e.g., a friendly customer service agent, a scientific expert).
            - **Tone**: The style or attitude in the language used by the LLM, such as formal, casual, or empathetic.
            - **Context**: The background information or setting provided to the LLM in a prompt to enhance response relevance and coherence.
            - **Template**: A structured format of a prompt with placeholders, used to generate similar responses across different contexts.
            - **Reference**: External or previously mentioned information that the LLM can use to support its responses.
            - **Candidates**: Multiple potential completions generated by the LLM from a single prompt, often ranked for quality or relevance or as an intermediate step before asking the LLM to select the best response based on some other criteria.
            - **Constraints**: Limitations or rules provided in the prompt to guide the model’s output (e.g., word limits, banned terms).
    - *Input Parameters:*
        - **Temperature**: Controls the randomness of the model’s output.
        - **Frequency Penalty**: Reduces the likelihood of repeating words.
        - **Presence Penalty**: Encourages the inclusion of new words.
        - **"Top-P Sampling"**: An alternative to the above penalties, where the response sequence selects the next word from the most probable set of outcomes, where the cumulative probability equals or exceeds a threshold *p*. This makes outputs more diverse while maintaining coherence.
    - *Workflows:*
        - *Web-Based:*
            - **Chat Agents**: Web-based interfaces or applications where users interact with LLMs in conversational form, such as [ChatGPT](https://chat.openai.com/).
            - **Playgrounds / Sandboxes**: [Interactive environments](https://platform.openai.com/playground) where developers can test and experiment with LLMs using various input/output examples, parameter adjustments, fine-tuning and training options.
        - *Programmatic:*
            - **Local API-Based Workflow**: A workflow using a programming language like R or Python to interact with LLMs hosted by a vendor on a local machine or server
            - **Embedded API-Based Workflow**

- *LLM Platforms / Tools:*
    - *Foundational Models*:
        - **GPT-4**: An LLM developed by OpenAI built with an emphasis on generalizabiltiy and "reasoning".
        - **Claude 3**: An LLM developed by Anthropic built with an emphasis on safety and [user alignment](https://en.wikipedia.org/wiki/AI_alignment).
        - **Llama 3**: A family of LLMs developed by Meta with an emphasis on open source access and academic NLP research.
        - **Gemini**: An LLM developed by Google DeepMind that incorporates reinforcement learning with an emphasis on multi-modality.
    - *Applications / Platforms*:
        - **ChatGPT**: Developed by OpenAI, an AI chatbot that uses GPT models for various natural language tasks like answering questions, writing text, and generating creative content.
        - **Ollama**: An open source community and self-titled tool for running language models locally on your device, designed to give users more control and privacy over the data that is processed without relying on cloud-based services.
        - **Azure OpenAI**: An integration of OpenAI’s GPT models into Microsoft's cloud computing environment (Azure).
        - **Hugging Face**: An open source community and repository where users can access pre-trained language models (including GPT-based ones) and use them for a variety of NLP tasks.

- *LLM Survey Applications:*
    - **AI Agent / Assistant**: A chatbot or virtual assistant designed to assist in collecting or analyzing survey data, often handling tasks like probing or follow-ups.
    - **Probing / Reprobing**: Techniques used in face-to-face (F2F) interviews, conversational interviews, and/or qualitative studies where the interviewer asks follow-up questions based on initial responses to gather more detailed information.
        - **Elaboration Probing**: The interviewer asks the respondent to provide more detail on a particular topic or answer.
        - **Quality / Relevance Probing**: The interviewer checks whether the response given is relevant or of sufficient quality.
        - **Confirmation Probing**: The interviewer asks for confirmation of a particular response to ensure accuracy.
    - **Synthetic Responses**: Artificial responses generated by an AI to mimic how a human might respond in a survey.
        - **Silicon Sample**: A fully AI-generated respondent used to simulate survey results and responses, typically for testing or research purposes.
    - **Tailoring / Personalization**: Individualizing survey questions or prompts based on information known about the respondent to increase relevance or engagement.
    - **Imputation**: The process of filling in missing data points in a dataset using AI models to predict what those values might be based on existing data.
    - **Retrodiction**: The task of using current data to infer past events or states, often using LLMs to recreate historical responses or trends.
    - **Nuisance Tasks**: Small, repetitive tasks like labeling axes in plots or adding metadata to survey questions, often very easily handled by LLMs to save time.

- *LLM Performance Concepts:*
    - **Model Evaluation / Validation**: The process of assessing how well a model performs across a broad range of generalized computing and "reasoning" tasks.
    - **Downstream Evaluation / Validation**: The process of assessing a model’s performance on real-world applications after initial training and testing. *In this course, we would like to stress that this is the kind of validation that most survey researchers should care about - NOT generalized model evaluation.*
    - **Metrics:**
        - **Intercoder Reliability**: A measure of consistency between different human coders or raters when interpreting or labeling text.
        - **Accuracy**: The percentage of correct predictions made by the model, out of all predictions made (assumes that the ground truth is available).
        - **F1-Score**: A harmonic mean of precision and recall, providing a balanced summary of both false positives and false negatives in a set of predictions.
        - **Processing Time / Latency**: The time it takes for the model to generate a response after receiving a prompt.
        - **Readability (Flesch-Kincaid)**: A measure of how easy a text is to read, based on sentence length and word complexity.
        - **Corrected Type-to-Token Ratio (CTTR)**: A measure of lexical diversity that accounts for the total number of unique words (types) in proportion to the total number of words (tokens).
        - **Lexical Diversity**: A measure of how varied the vocabulary used in a text is, reflecting the richness of the language.
        - (Specific to Synthetic Responses):
            - **Algorithmic Fidelity**: A measure of how varied the vocabulary used in a text is, reflecting the richness of the language.
            - **Backward Continuity**: Consistency in the model’s responses when considering the preceding context of a conversation or text generation.
            - **Forward Continuity**: The ability of a model to maintain coherence and consistency when generating subsequent text.
            - **Pattern Correspondence**: The extent to which AI-generated text follows recognizable patterns found in human communication.
        - **Sensitivity**: Whether and how an LLM's performance changes on as specific task based on small tweaks to the inputs or structure (bad!); for instance, [Bisbee et al. (2024)](https://www.cambridge.org/core/journals/political-analysis/article/synthetic-replacements-for-human-survey-data-the-perils-of-large-language-models/B92267DC26195C7F36E63EA04A47D2FE) find that synthetic responses to survey questions are sensitive to both minor changes in prompt texts as well as to the time period the quetsion is asked.
        - **Reliability**: The consistency of the model’s performance across different runs, data subsets, or tasks.
        - **ELO Ratings**: A system used to rank AI models based on their performance relative to each other, often used in competitive environments (e.g. the [Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/))

- *Other Gen AI Concepts:*
    - **Hallucination**: When a model generates plausible-sounding but incorrect or nonsensical information.
    - **Alignment**: Ensuring that an AI system’s outputs match the goals, values, or ethics of its users or developers.
    - **Bias**: When a model’s output reflects unfair or unbalanced treatment of certain groups or ideas due to biases in the training data.
    - **Toxicity**: The generation of harmful, offensive, or inappropriate content by a model, often an important consideration in content moderation.
    - **Model Drift**: When an AI model's performance degrades over time due to changes in the environment or data, requiring retraining or updates.
