import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# --- Configuration ---
INPUT_FILE = 'Input.xlsx'
STOP_WORDS_DIR = './'
POSITIVE_WORDS_FILE = 'positive-words.txt'
NEGATIVE_WORDS_FILE = 'negative-words.txt'
OUTPUT_FILE = 'Output.xlsx'
EXTRACTED_DIR = 'extracted_articles'

# --- Setup ---
def setup():
    """Download necessary NLTK data and create directories."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
    
    if not os.path.exists(EXTRACTED_DIR):
        os.makedirs(EXTRACTED_DIR)

# --- Data Extraction ---
def scrape_article(url):
    """Extract title and text from the given URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.find('h1')
        title_text = title.get_text(strip=True) if title else ""

        article_content = soup.find('div', class_='td-post-content')
        if not article_content:
            article_content = soup.find('div', class_='tdb-block-inner td-fix-index')
        
        if article_content:
            for tag in article_content(['script', 'style', 'header', 'footer', 'nav']):
                tag.decompose()
            content_text = article_content.get_text(separator='\n', strip=True)
        else:
            paragraphs = soup.find_all('p')
            content_text = '\n'.join([p.get_text(strip=True) for p in paragraphs])

        return title_text, content_text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None, None

# --- Text Analysis Helpers ---
def load_words(file_path, encoding='utf-8'):
    """Load words from a file into a set."""
    words = set()
    if not os.path.exists(file_path):
        return words
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            for line in f:
                word = line.split('|')[0].strip().lower()
                if word:
                    words.add(word)
    except Exception:
        try:
            with open(file_path, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    word = line.split('|')[0].strip().lower()
                    if word:
                        words.add(word)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return words

def count_syllables(word):
    """Count syllables in a word."""
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if not word: return 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count <= 0:
        count = 1
    return count

def analyze_text(text, stop_words, positive_words, negative_words):
    """Perform textual analysis and compute variables."""
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    num_words_total = len(words)
    
    cleaned_words = [word for word in words if word not in stop_words]
    num_cleaned_words = len(cleaned_words)
    
    pos_score = sum(1 for word in cleaned_words if word in positive_words)
    neg_score = sum(1 for word in cleaned_words if word in negative_words)
    
    polarity = (pos_score - neg_score) / ((pos_score + neg_score) + 0.000001)
    subjectivity = (pos_score + neg_score) / (num_cleaned_words + 0.000001)
    
    avg_sent_len = num_words_total / (num_sentences + 0.000001)
    
    complex_words = [word for word in words if count_syllables(word) > 2]
    num_complex = len(complex_words)
    pct_complex = num_complex / (num_words_total + 0.000001)
    
    fog_index = 0.4 * (avg_sent_len + pct_complex)
    
    word_count = num_cleaned_words
    
    total_syllables = sum(count_syllables(word) for word in words)
    syllables_per_word = total_syllables / (num_words_total + 0.000001)
    
    pronoun_pattern = re.compile(r'\b(I|we|my|ours|us)\b', re.I)
    matches = pronoun_pattern.findall(text)
    pronoun_count = len([m for m in matches if m != 'US'])
    
    avg_word_len = sum(len(word) for word in words) / (num_words_total + 0.000001)
    
    return {
        'POSITIVE SCORE': pos_score,
        'NEGATIVE SCORE': neg_score,
        'POLARITY SCORE': polarity,
        'SUBJECTIVITY SCORE': subjectivity,
        'AVG SENTENCE LENGTH': avg_sent_len,
        'PERCENTAGE OF COMPLEX WORDS': pct_complex,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_sent_len,
        'COMPLEX WORD COUNT': num_complex,
        'WORD COUNT': word_count,
        'SYLLABLE PER WORD': syllables_per_word,
        'PERSONAL PRONOUNS': pronoun_count,
        'AVG WORD LENGTH': avg_word_len
    }

# --- Main Execution ---
def main():
    print("Starting process...")
    setup()
    
    # Load stop words
    stop_words = set()
    for f in os.listdir(STOP_WORDS_DIR):
        if f.startswith('StopWords_'):
            stop_words.update(load_words(os.path.join(STOP_WORDS_DIR, f)))
    
    pos_words = load_words(POSITIVE_WORDS_FILE)
    neg_words = load_words(NEGATIVE_WORDS_FILE)
    
    input_df = pd.read_excel(INPUT_FILE)
    results = []
    
    print("Extracting and analyzing articles...")
    for _, row in input_df.iterrows():
        url_id, url = row['URL_ID'], row['URL']
        print(f"Processing {url_id}...")
        
        title, content = scrape_article(url)
        if title or content:
            full_text = (title + "\n\n" + content) if title else content
            # Save extracted text
            with open(os.path.join(EXTRACTED_DIR, f"{url_id}.txt"), 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            analysis = analyze_text(full_text, stop_words, pos_words, neg_words)
            res_row = row.to_dict()
            res_row.update(analysis)
            results.append(res_row)
        else:
            res_row = row.to_dict()
            for col in ['POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE', 
                        'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 
                        'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT', 
                        'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']:
                res_row[col] = 0
            results.append(res_row)
        
        time.sleep(0.5)
    
    output_df = pd.DataFrame(results)
    cols = ['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE', 
            'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 
            'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT', 
            'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']
    output_df = output_df[cols]
    output_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Process complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
