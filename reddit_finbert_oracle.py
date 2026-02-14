import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import math
import os

# ==============================
# CONFIG
# ==============================

REDDIT_URL = "https://www.reddit.com/r/cryptocurrency/hot.json?limit=50"

HEADERS = {
    "User-Agent": "windows:sentiment-oracle:v7.0 (by /u/ImpressiveParsnip927)"
}

ALPHA = 0.3  # smoothing factor

# ==============================
# LOAD FINBERT (ONLY ONCE)
# ==============================

print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()
print("FinBERT Loaded.\n")

# ==============================
# FILTER CONFIG
# ==============================

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

texts = []
weights = []
processed_posts = 0

print("Processing Reddit posts...\n")

for post in posts:
    title = post["data"]["title"]
    body = post["data"].get("selftext", "")
    upvotes = post["data"]["score"]

    full_text = f"{title} {body}"

    if contains_excluded(full_text):
        continue

    kw_count = keyword_count(full_text)
    if kw_count == 0:
        continue

    engagement_weight = math.log(upvotes + 1)
    keyword_weight = kw_count
    final_weight = engagement_weight * keyword_weight

    texts.append(full_text)
    weights.append(final_weight)
    processed_posts += 1

if len(texts) == 0:
    print("⚠️ No valid posts processed.")
    exit()

# ==============================
# BATCH FINBERT INFERENCE
# ==============================

inputs = tokenizer(
    texts,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)

with torch.no_grad():
    outputs = model(**inputs)

probs = F.softmax(outputs.logits, dim=1)

# ==============================
# WEIGHTED AGGREGATION
# ==============================

weighted_total = 0
weight_sum = 0

for i in range(len(texts)):
    negative = probs[i][0].item()
    positive = probs[i][2].item()

    normalized_score = (positive - negative + 1) * 50

    weighted_total += normalized_score * weights[i]
    weight_sum += weights[i]

raw_score = weighted_total / weight_sum

# ==============================
# SIGNAL INTELLIGENCE
# ==============================

previous_score = load_previous_score()
if previous_score is None:
    previous_score = raw_score

smoothed_score = ALPHA * raw_score + (1 - ALPHA) * previous_score
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
# OUTPUT
# ==============================

print("\n================ ORACLE OUTPUT ================")
print(f"Posts Analyzed: {processed_posts}")
print(f"Raw Sentiment Score: {round(raw_score, 2)}")
print(f"Smoothed Sentiment Score: {round(smoothed_score, 2)}")
print(f"Momentum: {round(momentum, 2)}")
print(f"Volatility: {round(volatility, 2)}")
print(f"Signal State: {signal}")
print("================================================\n")
