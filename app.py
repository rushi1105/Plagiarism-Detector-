from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")
DetectorFactory.seed = 0

app = Flask(__name__)

# Load your model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Preprocessing function to load data and get a sample
def preprocess_data(data_path, sample_size):
    data = pd.read_csv(data_path, low_memory=False)
    data = data.dropna(subset=['abstract']).reset_index(drop=True)
    data = data.sample(sample_size)[['abstract', 'cord_uid']]
    return data

# Function to create vectors from text using BERT model
def create_vector_from_text(tokenizer, model, text, MAX_LEN=510):
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=MAX_LEN)
    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    input_ids = results[0]
    attention_mask = [int(i > 0) for i in input_ids]
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits, encoded_layers = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, return_dict=False)

    vector = encoded_layers[12][0][0].detach().cpu().numpy()
    return vector

# Create a vector database from a set of abstracts
def create_vector_database(data):
    vectors = []
    source_data = data.abstract.values
    for text in tqdm(source_data):
        vector = create_vector_from_text(tokenizer, model, text)
        vectors.append(vector)

    data["vectors"] = vectors
    data["vectors"] = data["vectors"].apply(lambda emb: np.array(emb).reshape(1, -1))
    return data

# Translation function
def translate_text(text, text_lang, target_lang='en'):
    model_name = f"Helsinki-NLP/opus-mt-{text_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    formatted_text = f">>{text_lang}<< {text}"
    translation = model.generate(**tokenizer([formatted_text], return_tensors="pt", padding=True))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translation][0]
    return translated_text

# Process a document to generate vectors
def process_document(text):
    text_vect = create_vector_from_text(tokenizer, model, text)
    text_vect = np.array(text_vect).reshape(1, -1)
    return text_vect

# Check if similarity score indicates plagiarism
def is_plagiarism(similarity_score, plagiarism_threshold):
    return similarity_score >= plagiarism_threshold

# Candidate Languages
language_list = ['de', 'fr', 'el', 'ja', 'ru']

# Detect and translate the document if necessary
def check_incoming_document(incoming_document):
    text_lang = detect(incoming_document)
    if text_lang == 'en':
        return incoming_document
    elif text_lang not in language_list:
        return None
    else:
        return translate_text(incoming_document, text_lang)

# Run plagiarism analysis on a query text
def run_plagiarism_analysis(query_text, data, plagiarism_threshold=0.8):
    top_N = 3
    document_translation = check_incoming_document(query_text)

    if document_translation is None:
        return {"error": "Only the following languages are supported: English, French, Russian, German, Greek, and Japanese"}

    query_vect = process_document(document_translation)
    data["similarity"] = data["vectors"].apply(lambda x: cosine_similarity(query_vect, x)[0][0])
    similar_articles = data.sort_values(by='similarity', ascending=False)[0:top_N+1]
    formatted_result = similar_articles[["abstract", "cord_uid", "similarity"]].reset_index(drop=True)
    similarity_score = formatted_result.iloc[0]["similarity"]
    most_similar_article = formatted_result.iloc[0]["abstract"]
    is_plagiarism_bool = is_plagiarism(similarity_score, plagiarism_threshold)

    plagiarism_decision = {
        'similarity_score': similarity_score,
        'is_plagiarism': is_plagiarism_bool,
        'most_similar_article': most_similar_article,
        'article_submitted': query_text
    }

    return plagiarism_decision

# Load data into memory when the application starts
data_path = r"D:\RUSHIKESH SHIT\dataset\metadata.csv"  # Change this path accordingly
vector_database = create_vector_database(preprocess_data(data_path, 100))

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    incoming_text = request.json.get('text', '')
    analysis_result = run_plagiarism_analysis(incoming_text, vector_database, plagiarism_threshold=0.8)
    return jsonify(analysis_result)

if __name__ == '__main__':
    app.run(debug=True)
