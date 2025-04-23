from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter

extract = URLExtract()

def fetch_stats(selected_user, df):
    if df.empty:
        return 0, 0, 0, 0

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Make a copy and guarantee message is always a non-null str
    df = df.copy()
    df['message'] = df['message'].fillna('').astype(str)

    num_messages = df.shape[0]
    all_words    = [w for msg in df['message'] for w in msg.split()]
    words        = len(all_words)

    # Count media safely
    num_media = df['message'].str.contains('<Media omitted>', na=False).sum()

    # Extract links
    links = []
    for msg in df['message']:
        links.extend(extract.find_urls(msg))
    num_links = len(links)

    return num_messages, words, num_media, num_links


def create_wordcloud(selected_user, df):
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            stop_words = f.read().splitlines()
    except:
        return None

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df.copy()
    df['message'] = df['message'].fillna('').astype(str)

    temp = df[
        (df['user'] != 'group_notification') &
        (~df['message'].str.contains('<Media omitted>', na=False))
    ]
    filtered_text = ' '.join(temp['message'])

    if not filtered_text:
        return None

    return WordCloud(width=800, height=400, background_color='white')\
        .generate(filtered_text)


def most_common_words(selected_user, df):
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            stop_words = f.read().splitlines()
    except:
        return pd.DataFrame()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df.copy()
    df['message'] = df['message'].fillna('').astype(str)

    temp = df[
        (df['user'] != 'group_notification') &
        (~df['message'].str.contains('<Media omitted>', na=False))
    ]

    words = [
        word
        for msg in temp['message']
        for word in msg.lower().split()
        if word not in stop_words
    ]

    return pd.DataFrame(Counter(words).most_common(10),
                        columns=['word','count'])


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    if df.empty:
        return pd.DataFrame()
    try:
        return df.pivot_table(
            index='day_name',
            columns='period',
            values='message',
            aggfunc='count',
            fill_value=0
        )
    except KeyError:
        return pd.DataFrame()


def monthly_timeline(selected_user, df):
    """
    Returns a DataFrame with columns ['time','message'] where 'time' is
    'Month Year' (e.g. 'Jan 2025') and 'message' is the count of messages.
    """
    if df.empty:
        return pd.DataFrame(columns=['time','message'])

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # group by year & month
    timeline = (
        df
        .groupby(['year', 'month_num', 'month'])
        .count()['message']
        .reset_index()
        .sort_values(['year', 'month_num'])
    )

    # create a human‐readable time column
    timeline['time'] = timeline['month'] + ' ' + timeline['year'].astype(str)
    return timeline[['time', 'message']]


def daily_timeline(selected_user, df):
    """
    Returns a DataFrame with columns ['only_date','message'] where
    'only_date' is a datetime.date and 'message' is the count of messages.
    """
    if df.empty:
        return pd.DataFrame(columns=['only_date','message'])

    # filter by user if needed
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # group by the 'only_date' column produced in preprocessor.py
    daily = (
        df
        .groupby('only_date')
        .count()['message']
        .reset_index()
    )
    # rename so the first column matches what app.py expects
    daily.columns = ['only_date', 'message']
    return daily


def most_busy_users(df):
    counts = df['user'].value_counts().head(5)
    percentages = round((counts / counts.sum()) * 100, 2)
    return counts, pd.DataFrame({
        'User': counts.index,
        'Messages': counts.values,
        'Percentage': percentages.values
    })
