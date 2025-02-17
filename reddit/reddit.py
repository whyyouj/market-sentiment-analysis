import json
import os
import praw
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initalizing Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="Accessing Reddit threads",
    username=os.getenv("REDDIT_USERNAME"),
    passkey=os.getenv("REDDIT_PASSKEY"),
    check_for_async=False
)

# Gathering Reddit Posts
def collect_posts(subreddits, search_terms, dt_from, dt_to):
    posts_data = []
    stats = {
        'total_posts': 0,
        'posts_per_subreddit': {},
        'start_date': dt_from.strftime('%Y-%m-%d'),
        'end_date': dt_to.strftime('%Y-%m-%d')
    }
    '''
    # Note that this takes in one value, without the 'r/' prefix
    subreddit = reddit.subreddit(subreddits)
    threads = subreddit.hot(limit = 50)
    # Alternative to the code above
    # This can be used for the time filters too: "day","hour","month","week","year","all"
    # subreddit.top(time_filter=VALID_TIME_FILTERS[index], limit=(70 if int(index) == 0 else index + 1 * 50))
    
    for t in threads:
        # Pineed posts
        if t.stickied:
            continue
    '''
    
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        stats['posts_per_subreddit'][subreddit_name] = 0
        
        for term in search_terms:
            try:
                # Search for posts within time limit
                for submission in subreddit.search(term, sort='new'):
                    post_date = datetime.fromtimestamp(submission.created_utc)
                    
                    # Check if post is within our time limit
                    if (post_date >= dt_from) and (post_date <= dt_to):
                        post_data = {
                            'subreddit': subreddit_name,
                            'title': submission.title,
                            'text': submission.selftext,
                            'url': f"https://reddit.com{submission.permalink}",
                            'score': submission.score,
                            'created_utc': post_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'author': str(submission.author),
                            'num_comments': submission.num_comments,
                            'search_term': term
                        }
                        
                        posts_data.append(post_data)
                        stats['posts_per_subreddit'][subreddit_name] += 1
                        stats['total_posts'] += 1
                        
            except Exception as e:
                print(f"Error collecting data from r/{subreddit_name}: {str(e)}")
                continue
    
    return posts_data, stats

def save_to_json(data, filename):
    folder_path = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def print_summary(stats):
    print("\n=== Data Collection Summary ===")
    print(f"Time Period: {stats['end_date']} to {stats['start_date']}")
    print(f"\nTotal posts collected: {stats['total_posts']}")
    print("\nPosts per subreddit:")
    for subreddit, count in stats['posts_per_subreddit'].items():
        print(f"r/{subreddit}: {count} posts")

def main():
    # Define parameters
    subreddits = ['Gold', 'Economics', 'investing', 'wallstreetbets']
    search_terms = ['gold price', 'gold market', 'gold investment']
    dt_from = datetime(2025, 1, 1)
    dt_to = datetime(2025, 2, 1)
    
    print("Starting data collection...")
    print(f"Searching in subreddits: {', '.join(['r/' + s for s in subreddits])}")
    print(f"Search terms: {', '.join(search_terms)}")
    
    # Collect data
    posts_data, stats = collect_posts(subreddits, search_terms, dt_from, dt_to)
    
    # Save data
    timestamp_from = dt_from.strftime('%Y%m%d_%H%M%S')
    timestamp_to = dt_to.strftime('%Y%m%d_%H%M%S')
    filename = f'reddit_gold_data_{timestamp_from}_{timestamp_to}.json'
    save_to_json(posts_data, filename)
    
    # Print summary
    print(f"\nData saved to: {filename}")
    print_summary(stats)

if __name__ == "__main__":
    main()