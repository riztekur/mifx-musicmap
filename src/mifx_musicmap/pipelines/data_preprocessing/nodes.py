import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import scorecardpy as sc

def fill_missing_net_users(df):
    missing_count = df['net_user'].isna().cumsum()
    df.loc[df['net_user'].isna(), 'net_user'] = 'unknown_' + (missing_count).astype(str)
    return df

def make_age_category(df):
    df["age_category"] = pd.cut(
        df["age"],
        bins=[0, 2, 5, 13, 20, 40, 60, 100], 
        labels=["Infant", "Toddlers", "Children", "Teens", "Adults", "MiddleAged", "Senior"]
    )
    df["age_category"] = df["age_category"].cat.add_categories("Unknown")
    df["age_category"] = df["age_category"].fillna("Unknown")
    return df

def make_gender(df):
    df["gender"] = df["male"].replace(
        {
            1:"Male",
            0:"Female"
        }
    ).fillna("Unknown")
    df.drop(columns=["male"], inplace=True)
    return df

def fill_missing_good_country(df):
    df["good_country"] = df["good_country"].replace(
        {
            1:"True",
            0:"False"
        }
    ).fillna("Unknown")
    return df

def make_consistent_friend_cnt(df):
    df.loc[df['friend_cnt'] == 0, 'avg_friend_age'] = None
    df.loc[df['friend_cnt'] == 0, 'avg_friend_male'] = None
    df.loc[df['friend_cnt'] == 0, 'friend_country_cnt'] = 0
    df.loc[df['friend_cnt'] == 0, 'subscriber_friend_cnt'] = 0
    return df

def make_songsListened_monthly(df):
    df["songsListened_monthly"] = df["songsListened"] / (df["tenure"] + 1)
    return df

def make_social_activity(df):
    df['social_activity'] = df['posts'] + df['playlists'] + df['shouts']
    return df

def split_bot_users(df):
    q80 = df["songsListened_monthly"].quantile(0.8)
    human_users = df[df["songsListened_monthly"] <= q80]
    bot_users = df[df["songsListened_monthly"] > q80]
    return human_users, bot_users

def fix_missing_values(df):
    df = df.drop(columns=["net_user","age"])
    df = df.dropna(subset=["friend_cnt","friend_country_cnt","subscriber_friend_cnt","shouts","social_activity"])
    df['avg_friend_age'] = df['avg_friend_age'].fillna(df['avg_friend_age'].median())
    df['avg_friend_male'] = df['avg_friend_male'].fillna(df['avg_friend_male'].median())
    return df

def label_encode(df):
    label_encoders = {}
    for col in df.select_dtypes(include=[object]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df

def make_bins(data):
    bins = sc.woebin(data, y='adopter')
    return bins

def woe_transformer(data, bins):
    merged_dataset_woe = sc.woebin_ply(data, bins)
    return merged_dataset_woe

def split_dataset(df):
    train, test = train_test_split(df, stratify=df['adopter'], test_size=0.2, random_state=42)
    return train, test