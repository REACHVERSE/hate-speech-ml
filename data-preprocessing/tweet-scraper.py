import asyncio
import pandas as pd
import os
from twscrape import API
from twscrape.logger import set_log_level
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = 'raw-data/hate_speech_dataset_v2.csv'
SCRAPED_DATA_PATH = 'raw-data/scraped_tweets.csv'
DATA_DIR = 'raw-data'
ACCOUNTS_DB_PATH = "accounts.db"

MAX_TWEETS_TO_SCRAPE = 10000

# Load Twitter account credentials from environment variables
ACCOUNT_USERNAME = os.getenv("TWITTER_USERNAME")
ACCOUNT_PASSWORD = os.getenv("TWITTER_PASSWORD")
ACCOUNT_EMAIL = os.getenv("EMAIL_ADDRESS")
ACCOUNT_EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
ACCOUNT_COOKIES = os.getenv("ACCOUNT_COOKIES")


async def scrape_tweet_details(api: API, tweet_id: int, hate_label, topic_id):
    """
    Asynchronously scrapes details for a single tweet ID.
    Returns a dictionary with only Tweet, HateLabel, and TopicID data, or None if scraping fails.
    """
    try:
        tweet = await api.tweet_details(tweet_id)
        if tweet:
            return {
                'Tweet': tweet.rawContent,
                'HateLabel': hate_label,
                'TopicID': topic_id,
                'TweetID': tweet_id
            }
        else:
            print(f"INFO: Tweet {tweet_id} not found or unavailable.")
            return None
    except Exception as e:
        print(f"ERROR: Could not scrape tweet {tweet_id}. Reason: {e}")
        return None


async def main():
    """
    Main function to read tweet IDs, scrape them using twscrape concurrently,
    and save the results incrementally, with filtering by LangID and including TopicID.
    """
    set_log_level("WARNING")

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Input dataset not found at {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)

    required_cols = ['TweetID', 'HateLabel', 'LangID', 'TopicID']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns ({required_cols}) in the dataset.")
        return

    # Filter for English tweets (LangID == 1)
    df_filtered_lang = df[df['LangID'] == 1].copy()
    tweets_metadata = df_filtered_lang[['TweetID', 'HateLabel', 'TopicID']].dropna(subset=['TweetID']).copy()

    tweets_metadata['TweetID'] = pd.to_numeric(tweets_metadata['TweetID'], errors='coerce').astype('Int64')
    tweets_metadata.dropna(subset=['TweetID'], inplace=True)

    print(f"Found {len(tweets_metadata)} potential English (LangID=1) Tweet IDs to scrape in total from the dataset.")

    api = API(ACCOUNTS_DB_PATH)

    print("\nAttempting to add and log in with the provided account credentials...")
    try:
        await api.pool.add_account(
            ACCOUNT_USERNAME,
            ACCOUNT_PASSWORD,
            ACCOUNT_EMAIL,
            ACCOUNT_EMAIL_PASSWORD,
            cookies=ACCOUNT_COOKIES
        )
        await api.pool.login_all()
    except Exception as e:
        print(f"Error adding or logging in account: {e}")
        print("Please ensure your account credentials are correct and valid.")
        return

    if not await api.pool.accounts_info():
        print("\n" + "="*50)
        print("ERROR: No Twitter accounts are logged in successfully.")
        print("Please verify your account credentials and cookies.")
        print("="*50 + "\n")
        return
    else:
        print("Account(s) successfully added and logged in. Ready to scrape.")

    # Load already scraped Tweet IDs to enable resumption of scraping
    scraped_ids = set()
    if os.path.exists(SCRAPED_DATA_PATH):
        try:
            existing_df = pd.read_csv(SCRAPED_DATA_PATH, usecols=['TweetID'], dtype={'TweetID': 'Int64'})
            scraped_ids = set(existing_df['TweetID'].dropna().tolist())
            print(f"Loaded {len(scraped_ids)} previously scraped tweets from {SCRAPED_DATA_PATH}")
        except Exception as e:
            print(f"WARNING: Could not load existing scraped data from {SCRAPED_DATA_PATH}. Reason: {e}")
            scraped_ids = set()

    # Filter out already scraped tweets and limit to MAX_TWEETS_TO_SCRAPE
    tweets_to_scrape_df = tweets_metadata[
        ~tweets_metadata['TweetID'].isin(scraped_ids)
    ].head(MAX_TWEETS_TO_SCRAPE).copy()

    remaining_tweets_count = len(tweets_to_scrape_df)
    print(f"Found {remaining_tweets_count} unique English tweets remaining to scrape (limited to {MAX_TWEETS_TO_SCRAPE} for this run).")

    if remaining_tweets_count == 0:
        print("No new English tweets to scrape. All tweets from the input dataset appear to be scraped already or limit reached.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    tasks = []
    for index, row in tweets_to_scrape_df.iterrows():
        tweet_id = row['TweetID']
        hate_label = row['HateLabel']
        topic_id = row['TopicID']
        tasks.append(scrape_tweet_details(api, tweet_id, hate_label, topic_id))

    print(f"\nInitiating scraping for {len(tasks)} tweets concurrently...")
    all_results = await asyncio.gather(*tasks)
    successful_scrapes = [res for res in all_results if res is not None]

    if successful_scrapes:
        scraped_df = pd.DataFrame(successful_scrapes)

        # Ensure TweetID is integer type for resumption logic
        scraped_df['TweetID'] = pd.to_numeric(scraped_df['TweetID'], errors='coerce').astype('Int64')

        # Determine if header needs to be written for CSV (new file or empty file)
        write_header = not os.path.exists(SCRAPED_DATA_PATH) or os.stat(SCRAPED_DATA_PATH).st_size == 0

        scraped_df.to_csv(
            SCRAPED_DATA_PATH,
            mode='a', # Append mode to add new data to existing file
            index=False,
            encoding='utf-8',
            header=write_header
        )
        total_processed_in_this_run = len(scraped_df)
        print(f"All scraping tasks completed. Saved {total_processed_in_this_run} new tweets.")
    else:
        print("All scraping tasks completed with no new successful scrapes.")
        total_processed_in_this_run = 0

    print(f"\n--- Scraping session finished ---")
    print(f"Total new tweets successfully scraped in this run: {total_processed_in_this_run}")
    if os.path.exists(SCRAPED_DATA_PATH):
        try:
            final_df = pd.read_csv(SCRAPED_DATA_PATH)
            print(f"Total tweets in {SCRAPED_DATA_PATH}: {len(final_df)}")
        except Exception as e:
            print(f"Error reading final scraped data for count: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScraping interrupted by user. Progress saved up to the last completed batch.")
