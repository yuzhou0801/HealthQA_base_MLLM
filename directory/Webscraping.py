import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import json
from time import sleep
from random import uniform

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
REQUEST_INTERVAL = (1.0, 2.0)
MAX_ATTEMPTS = 3

PATTERN_CONFIG = {
    'symptoms': {
        'keywords': r'symptoms?|signs?|symptom|characteristics',
        'synonyms': ['may include', 'perform', 'performance'],
        'weight_threshold': 2
    },
    'treatments': {
        'keywords': r'treat(?:ments?|ing)|management|therapy|advice|therapies|care|medication|treatment',
        'synonyms': ['how to manage', 'what to do'],
        'weight_threshold': 2
    }
}


def enhanced_content_extractor(soup, mode):
    """Enhanced Content Extractor"""
    config = PATTERN_CONFIG[mode]
    content_blocks = []

    # Generating Matching Patterns
    combined_pattern = re.compile(
        r'(' + config['keywords'] + r')|(' + '|'.join(config['synonyms']) + r')',
        re.IGNORECASE
    )

    # Search for all relevant elements
    for element in soup.find_all(['h2', 'h3', 'div', 'section']):
        element_text = element.get_text(strip=True)

        # Calculate Match Score
        title_matches = len(re.findall(combined_pattern, element_text))
        content_score = 0

        # Analysis of follow-up
        next_element = element.next_sibling
        content_buffer = []
        while next_element and len(content_buffer) < 5:
            if next_element.name in ['h2', 'h3']:
                break

            # Collection of content texts
            if next_element.name == 'ul':
                content_buffer.extend([li.get_text(' ', strip=True) for li in next_element.find_all('li')])
            elif next_element.name == 'p':
                text = next_element.get_text(' ', strip=True)
                if re.search(combined_pattern, text):
                    content_score += 2
                content_buffer.append(text)

            next_element = next_element.next_sibling

        # Calculation of total score
        total_score = title_matches * 2 + content_score
        if total_score >= config['weight_threshold'] and content_buffer:
            content_blocks.append((total_score, content_buffer))

    # Return the best match
    if content_blocks:
        return max(content_blocks, key=lambda x: x[0])[1]
    return ['Information not available']


def fetch_medical_data(url):
    """Access to medical data"""
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            return {
                'symptoms': enhanced_content_extractor(soup, 'symptoms'),
                'treatments': enhanced_content_extractor(soup, 'treatments')
            }
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                print(f"请求失败 [{url}]: {str(e)}")
                return {'symptoms': ['Request error'], 'treatments': ['Request error']}
            sleep(2 ** attempt)


def process_diseases(input_file):
    """Processing of disease data"""
    try:
        df = pd.read_csv(input_file)
        required_cols = ['Disease', 'Link']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Input file missing required columns")
    except Exception as e:
        print(f"File read error: {str(e)}")
        return

    # Handling of each disease entry
    results = []
    total = len(df)
    for idx, row in df.iterrows():
        print(f"Processing [{idx + 1}/{total}]: {row['Disease']}")

        medical_info = fetch_medical_data(row['Link'])

        results.append({
            'Disease': row['Disease'],
            'Symptoms': medical_info['symptoms'],
            'Treatments': medical_info['treatments']
        })

        sleep(uniform(*REQUEST_INTERVAL))

    output_df = pd.DataFrame(results)

    output_df.to_csv('NHS_Data.csv', index=False)

    with open('NHS_Data.json', 'w') as f:
        json.dump(output_df.to_dict('records'), f, indent=2)

    print(f"\nProcessing complete! Generate {len(output_df)} rows")

if __name__ == "__main__":
    process_diseases('NHS_Full_Disease_Catalog.csv')
