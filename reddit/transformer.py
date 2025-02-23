import json
import os
from transformers import pipeline



def main():
    input_filename = "reddit_gold_data_20250101_000000_20250201_000000.json"
    results_folder = "results"
    output_filename = f"results_{input_filename}"
    output_filepath = os.path.join(results_folder, output_filename)
    
    # Load the Reddit dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", input_filename)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Initialize the sentiment analysis pipeline from huggingface
    sentiment_analyzer = pipeline("sentiment-analysis", model="minh21/XLNet-Reddit-Sentiment-Analysis")
    

    analysis_results = []
    
    # Process each post using the "text" field
    for post in data:
        text = post.get("text")
        if text and text.strip():
            result = sentiment_analyzer(text)
            analysis_results.append({
                "text": text,
                "url": post.get("url", "No URL provided"),
                "sentiment": result
            })
        else:
            analysis_results.append({
                "text": None,
                "url": post.get("url", "No URL provided"),
                "sentiment": "No valid text to analyze."
            })
    

    with open(output_filepath, "w", encoding="utf-8") as out_file:
        json.dump(analysis_results, out_file, indent=4, ensure_ascii=False)
    
    print(f"Analysis has been saved to {output_filepath}")

if __name__ == "__main__":
    main()
