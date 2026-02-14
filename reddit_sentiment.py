import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import math
import os

# ==============================
# CONFIG
# ==============================

REDDIT_URL = "https://www.reddit.com/r/cryptocurrency/hot.json?limit=50"

HEADERS = {
    "User-Agent": "windows:sentiment-oracle:v6.0 (by /u/ImpressiveParsnip927)"
}

ALPHA = 0.3  # smoothing factor

# ==============================
# NLP SETUP
# ==============================

analyzer = SentimentIntensityAnalyzer()

crypto_keywords = [
    "bitcoin", "btc", "ethereum", "eth",
    "crypto", "blockchain", "altcoin",
    "solana", "xrp", "market", "trading",
    "bull", "bear", "price", "defi"
]

exclude_phrases = [
    "daily discussion",
    "meme"
]

# ==============================
# HELPER FUNCTIONS
# ==============================

def keyword_count(text):
    text = text.lower()
    return sum(k in text for k in crypto_keywords)

def contains_excluded(text):
    text = text.lower()
    return any(p in text for p in exclude_phrases)

def load_previous_score():
    if os.path.exists("previous_score.txt"):
        with open("previous_score.txt", "r") as f:
            return float(f.read())
    return None

def save_score(score):
    with open("previous_score.txt", "w") as f:
        f.write(str(score))

# ==============================
# FETCH REDDIT DATA
# ==============================

response = requests.get(REDDIT_URL, headers=HEADERS)

if response.status_code != 200:
    print("❌ Reddit fetch failed:", response.status_code)
    exit()

data = response.json()
posts = data["data"]["children"]

# ==============================
# SENTIMENT + WEIGHTED AGGREGATION
# ==============================

weighted_total = 0
weight_sum = 0
processed_posts = 0

print("\n📊 Processing Reddit Sentiment (Balanced Mode)\n")

for post in posts:
    title = post["data"]["title"]
    body = post["data"].get("selftext", "")
    upvotes = post["data"]["score"]

    full_text = f"{title} {body}"

    # Basic noise filtering
    if contains_excluded(full_text):
        continue

    kw_count = keyword_count(full_text)
    if kw_count == 0:
        continue

    sentiment = analyzer.polarity_scores(full_text)
    compound = sentiment["compound"]

    normalized_score = (compound + 1) * 50  # -1..1 → 0..100

    engagement_weight = math.log(upvotes + 1)
    keyword_weight = kw_count

    final_weight = engagement_weight * keyword_weight

    weighted_total += normalized_score * final_weight
    weight_sum += final_weight
    processed_posts += 1

    print(f"Title: {title}")
    print(f"Upvotes: {upvotes} | Keywords: {kw_count}")
    print(f"Sentiment Score: {round(normalized_score, 2)}")
    print("-" * 80)

if weight_sum == 0:
    print("⚠️ No valid posts processed.")
    exit()

raw_score = weighted_total / weight_sum

# ==============================
# ADVANCED SIGNAL INTELLIGENCE
# ==============================

previous_score = load_previous_score()
if previous_score is None:
    previous_score = raw_score

# Exponential smoothing
smoothed_score = ALPHA * raw_score + (1 - ALPHA) * previous_score

# Momentum & volatility
momentum = smoothed_score - previous_score
volatility = abs(momentum)

save_score(smoothed_score)

# ==============================
# SIGNAL CLASSIFICATION
# ==============================

if smoothed_score > 75 and momentum > 5:
    signal = "🚀 STRONG BULLISH BREAKOUT"

elif smoothed_score < 30 and momentum < -5:
    signal = "📉 STRONG BEARISH BREAKDOWN"

elif volatility > 15:
    signal = "⚠️ HIGH VOLATILITY EVENT"

elif smoothed_score > 60:
    signal = "📈 BULLISH"

elif smoothed_score < 40:
    signal = "📉 BEARISH"

else:
    signal = "⚖️ NEUTRAL / STABLE"

# ==============================
# FINAL OUTPUT
# ==============================

print("\n================ ORACLE OUTPUT ================")
print(f"Posts Analyzed: {processed_posts}")
print(f"Raw Sentiment Score: {round(raw_score, 2)}")
print(f"Smoothed Sentiment Score: {round(smoothed_score, 2)}")
print(f"Momentum: {round(momentum, 2)}")
print(f"Volatility: {round(volatility, 2)}")
print(f"Signal State: {signal}")
print("================================================\n")
