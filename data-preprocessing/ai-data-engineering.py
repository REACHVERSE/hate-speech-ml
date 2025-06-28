import pandas as pd
import google.generativeai as genai
import numpy as np
import os
import time
from dotenv import load_dotenv

load_dotenv()

try:
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set the variable before running the script.")
    exit()

BATCH_SIZE = 25
API_DELAY_SECONDS = 4.1


def classify_text_batch_with_ai(text_batch):
    """
    Classifies a BATCH of texts in a single API call.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')

    numbered_texts = "\n".join([f'{i+1}. "{text}"' for i, text in enumerate(text_batch)])

    prompt = f"""
    Analyze each of the following texts and return ONLY the two comma-separated numerical IDs (TopicID,HateLabel) for EACH text. Each result must be on a new line.

    - TopicID Categories: 0=Religion, 1=Gender, 2=Race, 3=Politics, 4=Sports.
    - HateLabel Categories: 0=Normal, 1=Offensive, 2=Hate.
    - If you are uncertain for any text, use 0,0 for that line.
    - The output MUST have the exact same number of lines as the input texts.

    Example Response for 3 input texts:
    2,1
    0,0
    3,2

    --- START OF TEXTS ---
    {numbered_texts}
    --- END OF TEXTS ---
    """

    try:
        response = model.generate_content(prompt)
        results = []
        lines = response.text.strip().split('\n')
        for line in lines:
            try:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    topic_id = int(parts[0].strip())
                    hate_label = int(parts[1].strip())
                    if topic_id in range(5) and hate_label in range(3):
                        results.append((topic_id, hate_label))
                    else:
                        results.append((0, 0))
                else:
                    results.append((0, 0))
            except ValueError:
                results.append((0, 0))

        while len(results) < len(text_batch):
            results.append((0, 0))

        return results[:len(text_batch)]

    except Exception as e:
        print(f"Warning: AI batch API call failed. Error: {e}. Defaulting all texts in this batch.")
        return [(0, 0)] * len(text_batch)

def identify_text_column(df):
    """
    Identifies the primary text column in a DataFrame from a list of common names.
    """
    common_text_column_names = ['Text', 'Tweet', 'Review', 'Comment', 'Speech', 'text']
    for col in common_text_column_names:
        if col in df.columns:
            return col
    return None

def process_dataframe_with_ai_batched(df, df_name):
    """
    Processes a dataframe by grouping rows into batches, applying AI
    classification to each batch, and showing progress.
    """
    text_col = identify_text_column(df)
    if not text_col:
        print(f"Warning: No text column found in {df_name}. Skipping this file.")
        return pd.DataFrame(columns=['Text', 'TopicID', 'HateLabel'])

    print(f"Processing {df_name} using AI. Identified text column: '{text_col}'")

    clean_df = df.dropna(subset=[text_col]).copy()
    clean_df = clean_df[clean_df[text_col].astype(str).str.strip() != '']

    if clean_df.empty:
        print(f"Info: No valid text content found in {df_name} after cleaning.")
        return pd.DataFrame(columns=['Text', 'TopicID', 'HateLabel'])

    total_rows = len(clean_df)
    num_api_calls = -(-total_rows // BATCH_SIZE)
    estimated_minutes = (num_api_calls * API_DELAY_SECONDS) / 60
    print(f"Applying AI classification to {total_rows} rows from {df_name} in batches of {BATCH_SIZE}.")
    print(f"This will make {num_api_calls} API calls and take approximately {estimated_minutes:.1f} minutes.")

    all_results = []

    for i in range(0, total_rows, BATCH_SIZE):
        batch_df = clean_df.iloc[i:i+BATCH_SIZE]
        text_batch = batch_df[text_col].tolist()

        classified_results_tuples = classify_text_batch_with_ai(text_batch)

        for j, text_content in enumerate(text_batch):
            topic_id, hate_label = classified_results_tuples[j]
            all_results.append({
                'Text': text_content,
                'TopicID': topic_id,
                'HateLabel': hate_label
            })

        print(f"    ...processed batch ending at row {i + len(text_batch)} of {total_rows}.")

        # Sleep between batch API calls to respect the rate limit.
        if i + BATCH_SIZE < total_rows:
            time.sleep(API_DELAY_SECONDS)

    print(f"Finished processing {total_rows} rows from {df_name}.")

    if not all_results:
        return pd.DataFrame(columns=['Text', 'TopicID', 'HateLabel'])

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    all_dfs = []

    try:
        df_scraped = pd.read_csv('raw-data/scraped_tweets.csv')
        if 'Tweet' in df_scraped.columns:
            df_scraped = df_scraped.rename(columns={'Tweet': 'Text'})

        if all(col in df_scraped.columns for col in ['Text', 'TopicID', 'HateLabel']):
            all_dfs.append(df_scraped[['Text', 'TopicID', 'HateLabel']])
            print("Successfully loaded and formatted 'scraped_tweets.csv'.")
        else:
            print("Warning: 'scraped_tweets.csv' is missing required columns (Text, TopicID, HateLabel). Skipping.")
    except FileNotFoundError:
        print("Info: 'raw-data/scraped_tweets.csv' not found, skipping.")
    except Exception as e:
        print(f"Error loading 'scraped_tweets.csv': {e}. Skipping.")


    try:
        df_modified = pd.read_excel('raw-data/modified_hate_speech_dataset.xlsx')
        df_modified = df_modified.head(1000)
        processed_modified = process_dataframe_with_ai_batched(df_modified, 'modified_hate_speech_dataset.xlsx')
        all_dfs.append(processed_modified)
    except FileNotFoundError:
        print("Info: 'raw-data/modified_hate_speech_dataset.xlsx' not found, skipping.")
    except Exception as e:
        print(f"Error loading 'modified_hate_speech_dataset.xlsx': {e}. Skipping.")


    try:
        df_twitter = pd.read_csv('raw-data/twitter_parsed_dataset.csv')
        df_twitter = df_twitter.head(1000)
        processed_twitter = process_dataframe_with_ai_batched(df_twitter, 'twitter_parsed_dataset.csv')
        all_dfs.append(processed_twitter)
    except FileNotFoundError:
        print("Info: 'raw-data/twitter_parsed_dataset.csv' not found, skipping.")
    except Exception as e:
        print(f"Error loading 'twitter_parsed_dataset.csv': {e}. Skipping.")

    if not all_dfs:
        print("\nError: No data was successfully loaded or processed. Exiting.")
        exit()

    all_dfs = [df for df in all_dfs if not df.empty]
    if not all_dfs:
        print("\nError: All datasets were empty after processing. Exiting.")
        exit()

    print("\nConsolidating all data...")
    combined_df = pd.concat(all_dfs, ignore_index=True)

    combined_df['Text'] = combined_df['Text'].astype(str).str.strip().str.lower()
    combined_df.replace('', np.nan, inplace=True)
    combined_df.dropna(subset=['Text'], inplace=True)

    initial_rows = len(combined_df)
    combined_df.drop_duplicates(subset=['Text'], keep='first', inplace=True)
    print(f"Removed {initial_rows - len(combined_df)} duplicate text entries.")

    combined_df['TopicID'] = pd.to_numeric(combined_df['TopicID'], errors='coerce').fillna(0).astype(int)
    combined_df['HateLabel'] = pd.to_numeric(combined_df['HateLabel'], errors='coerce').fillna(0).astype(int)

    output_filename = 'processed-data/processed_dataset.csv'
    combined_df.to_csv(output_filename, index=False)

    print(f"\nProcessing complete. The final dataset is saved as '{output_filename}'.")
    print("\n--- Final Dataset Statistics ---")
    print(f"Total unique entries: {len(combined_df)}")

    print("\nHateLabel Distribution (0=Normal, 1=Offensive, 2=Hate):")
    print(combined_df['HateLabel'].value_counts())

    print("\nTopicID Distribution (0=Religion, 1=Gender, 2=Race, 3=Politics, 4=Sports):")
    print(combined_df['TopicID'].value_counts())
