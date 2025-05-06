import streamlit as st
# ── This must be the first Streamlit call ────────────────────
st.set_page_config("Reddit Sentiment Screener", "📈", layout="wide")

from openai import OpenAI
import praw, yfinance as yf, pandas as pd
import datetime, html, json, re, requests, io, certifi
from typing import List, Dict

# ════════════════════════════════════════════════════════
# 0 ▸ Symbol master  (download → fallback on error)
# ════════════════════════════════════════════════════════
FALLBACK_CSV = """Symbol,Name
AAPL,Apple Inc.
MSFT,Microsoft Corporation
TSLA,Tesla Inc.
GOOGL,Alphabet Inc. Class A
AMZN,Amazon.com Inc.
META,Meta Platforms Inc.
NVDA,NVIDIA Corporation
"""

@st.cache_data(ttl=86_400)
def load_symbol_master() -> pd.DataFrame:
    url = (
        "https://raw.githubusercontent.com/datasets/nasdaq-listings/"
        "master/data/nasdaq-listed-symbols.csv"
    )
    try:
        resp = requests.get(url, timeout=10, verify=certifi.where())
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), usecols=["Symbol", "Security Name"])
        df.rename(columns={"Security Name": "Name"}, inplace=True)
    except Exception:
        st.warning("⚠️ Couldn’t download full ticker list; using fallback symbols.")
        df = pd.read_csv(io.StringIO(FALLBACK_CSV))
    df["Label"] = df["Symbol"] + " — " + df["Name"]
    return df

SYMBOL_DF = load_symbol_master()

# ════════════════════════════════════════════════════════
# 1 ▸ Clients & constants
# ════════════════════════════════════════════════════════
ai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
reddit = praw.Reddit(
    client_id     = st.secrets["REDDIT_CLIENT_ID"],
    client_secret = st.secrets["REDDIT_CLIENT_SECRET"],
    user_agent    = st.secrets["REDDIT_USER_AGENT"],
)

SLACK_URL  = st.secrets.get("SLACK_WEBHOOK_URL", "")
SUBS       = "stocks+wallstreetbets"
MIN_POSTS  = 3          # ← lowered so GPT runs with 3-9 posts too
MAX_FETCH  = 300

# ════════════════════════════════════════════════════════
# 2 ▸ Helpers
# ════════════════════════════════════════════════════════
@st.cache_data(ttl=900)
def get_market(t: str) -> Dict:
    info = yf.Ticker(t).info
    price, prev = info["regularMarketPrice"], info["regularMarketPreviousClose"]
    pct = (price - prev) / prev * 100 if prev else 0
    return {
        "name":   info.get("shortName", t),
        "price":  f"${price:,.2f}",
        "change": f"{pct:+.2f} %",
        "range":  f"{info.get('fiftyTwoWeekLow','?')} – {info.get('fiftyTwoWeekHigh','?')}",
        "cap":    f"${info.get('marketCap',0):,}",
        "pe":     info.get("trailingPE", "—"),
        "beta":   info.get("beta", "—"),
        "vol":    f"{info.get('volume',0):,}",
        "avgVol": f"{info.get('averageVolume',0):,}",
    }

@st.cache_data(ttl=900)
def get_reddit(ticker: str) -> pd.DataFrame:
    """Fetch posts whose title contains $TICKER or bare TICKER word."""
    pat = re.compile(rf"(\${ticker}\b|\b{ticker}\b)", re.IGNORECASE)
    rows, fetched = [], 0
    for s in reddit.subreddit(SUBS).search(f"${ticker}", sort="top",
                                           time_filter="week", limit=MAX_FETCH):
        fetched += 1
        if not pat.search(s.title):
            continue
        age_h = int((datetime.datetime.utcnow() -
                     datetime.datetime.utcfromtimestamp(s.created_utc)).total_seconds() // 3600)
        rows.append({
            "score": s.score,
            "title": html.unescape(s.title),
            "url":   f"https://www.reddit.com{s.permalink}",
            "sub":   s.subreddit.display_name,
            "age_h": age_h,
        })
        if len(rows) >= 10:    # we only ever need up to 10 for GPT
            break
    return pd.DataFrame(rows).sort_values("score", ascending=False)

def gpt_summary(titles: List[str]) -> Dict:
    if len(titles) < MIN_POSTS:
        return {"error": f"Only {len(titles)} matching Reddit posts found."}

    use_n = min(len(titles), 10)      # send up to 10 (but at least 3)
    prompt = (f"Return ONLY a JSON object with keys:\n"
              f"sentiment   (Bullish|Bearish|Neutral),\n"
              f"highlights  (3-5 bullets, ≤12 words each),\n"
              f"risks       (2-3 bullets, ≤12 words each).\n"
              f"Use exactly {use_n} Reddit post titles provided.")
    resp = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":prompt},
                  {"role":"user","content":"\n".join(titles[:use_n])}],
        temperature=0.2, max_tokens=220,
        response_format={"type":"json_object"},
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else {"error": "Un-parsable GPT JSON."}

def slack_post(msg: str) -> bool:
    if not SLACK_URL: return False
    try:
        requests.post(SLACK_URL, json={"text": msg}, timeout=4)
        return True
    except Exception:
        return False

# ════════════════════════════════════════════════════════
# 3 ▸ UI
# ════════════════════════════════════════════════════════
st.header("Reddit Sentiment Stock Screener")

labels  = SYMBOL_DF["Label"].tolist()
default = labels.index("AAPL — Apple Inc.") if "AAPL — Apple Inc." in labels else 0
choice  = st.selectbox("Search company or ticker", labels, index=default)
ticker  = choice.split(" — ")[0]

run   = st.button("Refresh")
tsbox = st.empty()

if run:
    tsbox.markdown(f"*UTC updated*: {datetime.datetime.utcnow():%Y-%m-%d %H:%M:%S}")

    with st.spinner("Market data…"):  market = get_market(ticker)
    with st.spinner("Reddit posts…"): df     = get_reddit(ticker)
    with st.spinner("GPT analysis…"): llm    = gpt_summary(df["title"].tolist())

    col1, col2 = st.columns(2)

    # ── Column 1: Market Snapshot ─────────────────────────
    with col1:
        st.subheader("Market Snapshot")
        st.markdown(f"### {market['name']} ({ticker})")
        st.markdown(f"**{market['price']}** &nbsp; {market['change']}")
        st.markdown(
            f"*52-wk range*: {market['range']}  \n"
            f"*Market cap*: {market['cap']}  \n"
            f"*P/E*: {market['pe']} &nbsp; | &nbsp; *Beta*: {market['beta']}  \n"
            f"*Vol / Avg*: {market['vol']} / {market['avgVol']}"
        )

    # ── Column 2: AI Insight ─────────────────────────────
    with col2:
        st.subheader("AI Insight")
        if "error" in llm:
            st.error(llm["error"])
        else:
            pill = {"Bullish":"#2ecc71","Bearish":"#e74c3c","Neutral":"#f39c12"}.get(
                llm["sentiment"], "#666")
            st.markdown(
                f"<span style='background:{pill};padding:4px 12px;border-radius:12px;color:#fff'>"
                f"{llm['sentiment']}</span>", unsafe_allow_html=True
            )
            st.markdown(f"*Posts analysed*: {len(df)}")
            st.write("**Highlights**")
            st.markdown("\n".join(f"- {h}" for h in llm["highlights"]))
            st.write("**Risks**")
            st.markdown("\n".join(f"- {r}" for r in llm["risks"]))

    # ── Reddit table ─────────────────────────────────────
    st.subheader("Top Reddit Posts (last week)")
    view = df.reset_index(drop=True)
    st.dataframe(
        view.assign(Link=view["url"])[["score","title","Link","sub","age_h"]]
        .rename(columns={"score":"Score","title":"Title","sub":"Sub","age_h":"Age (h)"}),
        hide_index=True,
        column_config={
            "Link":  st.column_config.LinkColumn("Link"),
            "Title": st.column_config.Column(width="large")
        }
    )

    # ── Footer ───────────────────────────────────────────
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, f"{ticker}_reddit.csv", "text/csv", use_container_width=True)
    if SLACK_URL and st.button("Share to Slack"):
        ok = slack_post(f"*{ticker}* • {market['price']} {market['change']} • {llm.get('sentiment','?')}")
        st.success("Sent!") if ok else st.error("Slack error")
    st.caption("Not financial advice.")
else:
    st.info("Pick a company and click **Refresh**.")
