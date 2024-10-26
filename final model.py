import pandas as pd
from tqdm import tqdm
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

DetectorFactory.seed = 0

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)


# Preprocessing function to load data and get a sample
def preprocess_data(data_path, sample_size):
    data = pd.read_csv(data_path, low_memory=False)
    data = data.dropna(subset=['abstract']).reset_index(drop=True)
    data = data.sample(sample_size)[['abstract', 'cord_uid']]
    return data

# Define the data path and sample size
data_path = r"D:\RUSHIKESH SHIT\dataset\metadata.csv"
source_data = preprocess_data(data_path, 100)

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

# Function to create a vector database from a set of abstracts
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
        print("Only the following languages are supported: English, French, Russian, German, Greek, and Japanese")
        exit(-1)
    else:
        query_vect = process_document(document_translation)
        data["similarity"] = data["vectors"].apply(lambda x: cosine_similarity(query_vect, x)[0][0])
        similar_articles = data.sort_values(by='similarity', ascending=False)[0:top_N+1]
        formated_result = similar_articles[["abstract", "cord_uid", "similarity"]].reset_index(drop=True)
        similarity_score = formated_result.iloc[0]["similarity"]
        most_similar_article = formated_result.iloc[0]["abstract"]
        is_plagiarism_bool = is_plagiarism(similarity_score, plagiarism_threshold)
        
        plagiarism_decision = {
            'similarity_score': similarity_score,
            'is_plagiarism': is_plagiarism_bool,
            'most_similar_article': most_similar_article,
            'article_submitted': query_text
        }
        
        return plagiarism_decision

# Format and print the plagiarism result
def print_plagiarism_result(analysis_result):
    print("\nPlagiarism Detection Result")
    print("=" * 30)
    print(f"Similarity Score: {analysis_result['similarity_score']:.4f}")
    print(f"Is Plagiarism: {'Yes' if analysis_result['is_plagiarism'] else 'No'}")
    print("\nMost Similar Article:")
    print(analysis_result['most_similar_article'])
    print("\nArticle Submitted:")
    print(analysis_result['article_submitted'])
    print("=" * 30)

# Example usage
new_incoming_text = source_data.iloc[0]['abstract']
vector_database = create_vector_database(source_data)

# Run plagiarism detection for an existing article in the database
analysis_result = run_plagiarism_analysis(new_incoming_text, vector_database, plagiarism_threshold=0.8)
print_plagiarism_result(analysis_result)

# Check plagiarism for French and German articles
french_article_to_check = """
Les Réseaux d’Innovation et de Transfert Agricole (RITA) ont été créés en 2011 ...
"""
german_article_to_check = """
Derzeit ist eine Reihe strukturell und funktionell unterschiedlicher temperaturempfindlicher Elemente....
"""

# Run plagiarism check on French article
french_analysis_result = run_plagiarism_analysis(french_article_to_check, vector_database, plagiarism_threshold=0.8)
print_plagiarism_result(french_analysis_result)

# Run plagiarism check on German article
german_analysis_result = run_plagiarism_analysis(german_article_to_check, vector_database, plagiarism_threshold=0.8)
print_plagiarism_result(german_analysis_result)
