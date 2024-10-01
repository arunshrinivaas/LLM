import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv('your_dataset.csv')

# Ensure the columns are present
if not all(col in data.columns for col in ['Report Name', 'History', 'Observation']):
    raise ValueError("Dataset must contain 'Report Name', 'History', and 'Observation' columns.")

# Create a new column for training text
data['input_text'] = data['Report Name'] + ' ' + data['History'] + ' ' + data['Observation']

# Convert the DataFrame to a Dataset
dataset = Dataset.from_pandas(data[['input_text']])

# Split the dataset into train and eval sets (300 train, 30 eval)
train_test_split = dataset.train_test_split(test_size=30, shuffle=True, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Step 2: Choose the model and tokenizer
model_name = 'gemma-2b-it'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=512)

# Tokenizing the datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask'])
tokenized_eval.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Step 3: Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    evaluation_strategy='epoch',     # Evaluate every epoch
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=2,   # Batch size for training
    per_device_eval_batch_size=2,    # Batch size for evaluation
    num_train_epochs=3,               # Number of training epochs
    weight_decay=0.01,                # Strength of weight decay
    logging_dir='./logs',             # Directory for storing logs
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# Step 4: Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

print("Fine-tuning complete and model saved.")

# Step 5: Model Evaluation (using ROUGE and Perplexity)
def evaluate_model(eval_dataset):
    # Generate impressions for the evaluation set
    eval_texts = eval_dataset['input_text']
    inputs = tokenizer(eval_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Here you can compute ROUGE and Perplexity
    # Example for Perplexity
    perplexity = torch.exp(model(input_ids=inputs['input_ids'], labels=inputs['input_ids']).loss)
    print(f"Perplexity: {perplexity.item()}")

    return generated_texts

# Evaluate the model on the evaluation dataset
generated_impressions = evaluate_model(tokenized_eval)

# Step 6: Text Analysis (remove stop words, stemming, and lemmatization)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Download stopwords if not already done
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Function for text preprocessing
def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    words = [stemmer.stem(word) for word in words]  # Apply stemming
    words = [lemmatizer.lemmatize(word) for word in words]  # Apply lemmatization
    return ' '.join(words)

# Apply preprocessing to the entire dataset
data['processed_text'] = data['input_text'].apply(preprocess_text)

# Step 7: Generate Embeddings and Identify Top 100 Word Pairs
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['processed_text'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Identify top 100 pairs based on similarity
similar_pairs = np.argpartition(cosine_sim.flatten(), -100)[-100:]
top_pairs = [(i, j) for i in range(len(data)) for j in range(len(data)) if cosine_sim[i][j] > 0.5]

# Step 8: Visualization of Top 100 Word Pairs
top_words = [(data.iloc[i]['input_text'], data.iloc[j]['input_text'], cosine_sim[i][j]) for i, j in top_pairs]
top_words.sort(key=lambda x: x[2], reverse=True)

# Create a DataFrame for visualization
top_words_df = pd.DataFrame(top_words, columns=['Word 1', 'Word 2', 'Similarity'])
top_words_df = top_words_df.head(100)  # Take top 100 pairs

# Visualizing the top pairs
plt.figure(figsize=(10, 8))
plt.barh(top_words_df['Word 1'], top_words_df['Similarity'], color='skyblue')
plt.xlabel('Similarity')
plt.title('Top 100 Similar Word Pairs')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest on top
plt.show()
