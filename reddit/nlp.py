import os
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Uncomment this if you have never downloaded NLTK resources before
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("vader_lexicon")

def preprocess_text(text):
    # Tokenizee, stemer and lemmatizer
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    

    processed_tokens = []
    for token in tokens:
        stemmed = stemmer.stem(token)
        lemmatized = lemmatizer.lemmatize(stemmed)
        processed_tokens.append(lemmatized)
    
    # Join tokens back into a string
    return " ".join(processed_tokens)

def main():
    input_filename = "reddit_gold_data_20250101_000000_20250201_000000.json"
    results_folder = "results"
    # os.makedirs(results_folder, exist_ok=True)
    
    output_filename = f"nlp_results_{input_filename}"
    output_filepath = os.path.join(results_folder, output_filename)
    
    # Loading the Reddit dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", input_filename)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        

    sia = SentimentIntensityAnalyzer()
    

    analysis_results = []
    
    # Process each record in the "text" field
    for post in data:
        original_text = post.get("text")
        if original_text and original_text.strip():
            
            processed_text = preprocess_text(original_text)
            
            sentiment_scores = sia.polarity_scores(processed_text)
            analysis_results.append({
                "url": post.get("url", "No URL provided"),
                "original_text": original_text,
                "processed_text": processed_text,
                "sentiment": sentiment_scores
            })
        else: # If no text (data scraped might contain images only)
            analysis_results.append({
                "url": post.get("url", "No URL provided"),
                "original_text": None,
                "processed_text": None,
                "sentiment": "No valid text to analyze."
            })
    

    with open(output_filepath, "w", encoding="utf-8") as outfile:
        json.dump(analysis_results, outfile, indent=4, ensure_ascii=False)
    
    print(f"Analysis has been saved to {output_filepath}")

if __name__ == "__main__":
    main()
