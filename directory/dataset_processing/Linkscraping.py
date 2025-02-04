from bs4 import BeautifulSoup
import requests
import pandas as pd
import re


def main():
    url = "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z"
    response = requests.get(url)
    response.raise_for_status()
    parsed_code = BeautifulSoup(response.text, "html.parser")

    names = []
    links = []

    # Grab All Alphabets Module with CSS Selector
    letter_sections = parsed_code.select('div[class*="az_list_indivisual"]')

    for section in letter_sections:
        # Locate a specific list of diseases in each alphabetical module
        disease_items = section.find_all('li')

        for item in disease_items:
            link_tag = item.find('a', href=True)
            if link_tag:
                # Cleaning data
                name = re.sub(r'\s+', ' ', link_tag.text).strip()
                href = link_tag['href']

                # Build the full link
                if href.startswith('/'):
                    full_url = f'https://www.nhsinform.scot{href}'
                else:
                    full_url = href

                names.append(name)
                links.append(full_url)

    # Creating data frames and verifying data integrity
    if len(names) != len(links):
        raise ValueError(f"Data inconsistencies: number of names({len(names)}) ≠ Number of links({len(links)})")

    disease_data = pd.DataFrame({
        'Disease_ID': range(1, len(names) + 1),
        'Disease_Name': names,
        'Disease_Link': links
    })

    # Save data and output statistics
    disease_data.to_csv('NHS_Full_Disease_Catalog.csv', index=False)
    print(f"Grab {len(disease_data)} disease data")
    print("Sample data:")
    print(disease_data.head())


if __name__ == "__main__":
    main()
