import dash
from dash import dcc, html, dash_table
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime, timedelta
from google.cloud import bigquery
import os

# ✅ Set GCP credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/gcp_key.json"

# ✅ Load from BigQuery
def load_sentiment_data():
    client = bigquery.Client()
    query = """
        SELECT publishedAt, title, sentiment, topics, text, source, url
        FROM `steam-state-459919-s3.news_dashboard.sentiment`
        ORDER BY publishedAt DESC
        LIMIT 200
    """
    df = client.query(query).to_dataframe()
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    df["date"] = df["publishedAt"].dt.date
    df["hour"] = df["publishedAt"].dt.hour
    if isinstance(df["topics"].iloc[0], str):
        df["topics"] = df["topics"].apply(eval)
    return df

df = load_sentiment_data()

# ----- Filters -----
last_7_days = datetime.now().date() - timedelta(days=7)
df = df[df["date"] >= last_7_days]

date_from = df["date"].min().strftime("%B %d, %Y")
date_to = df["date"].max().strftime("%B %d, %Y")

# ----- Visualizations -----
sentiment_counts = df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["sentiment", "count"]
fig_pie = px.pie(sentiment_counts, names="sentiment", values="count", title="🧠 Sentiment Distribution")

text = " ".join(df["text"])
wc = WordCloud(width=800, height=400, background_color="white").generate(text)
buf = BytesIO()
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(buf, format="png")
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode()

sentiment_trend = df.groupby(["date", "sentiment"]).size().reset_index(name="count")
fig_line = px.line(sentiment_trend, x="date", y="count", color="sentiment", markers=True, title="📈 Sentiment Trend by Day")

source_sentiment = df.groupby(["source", "sentiment"]).size().reset_index(name="count")
fig_bar = px.bar(source_sentiment, x="source", y="count", color="sentiment", title="🗞️ Sentiment by News Source", barmode="group")

hourly_sentiment = df.groupby(["hour", "sentiment"]).size().reset_index(name="count")
fig_hour = px.bar(hourly_sentiment, x="hour", y="count", color="sentiment", title="⏰ Sentiment by Hour", barmode="group")

topic_counts = df.explode("topics")["topics"].value_counts().reset_index()
topic_counts.columns = ["Topic", "Articles"]
fig_topics = px.bar(topic_counts, x="Topic", y="Articles", title="🧠 Article Count by Topic", color="Topic")

df["title"] = df.apply(lambda row: f"[{row['title']}]({row['url']})", axis=1)
df["topics_display"] = df["topics"].apply(lambda x: ", ".join(x))
df_sorted = df.sort_values("publishedAt", ascending=False)

# ----- Dash App -----
app = dash.Dash(__name__)
app.title = "🧠 Tech News Sentiment"

def _card(content, center=False):
    return html.Div(style={
        "backgroundColor": "white",
        "padding": "20px",
        "borderRadius": "10px",
        "boxShadow": "0 0 10px rgba(0,0,0,0.1)",
        "marginBottom": "30px",
        "textAlign": "center" if center else "left"
    }, children=content)

app.layout = html.Div(style={
    "fontFamily": "Segoe UI, sans-serif",
    "backgroundColor": "#f4f6f9",
    "padding": "30px"
}, children=[
    html.H1("🧠 Tech News Sentiment Dashboard", style={"textAlign": "center", "color": "#2c3e50"}),
    html.H3(f"🗓️ Showing articles from {date_from} → {date_to}", style={"textAlign": "center", "color": "#444", "marginBottom": "30px"}),

    _card([dcc.Graph(figure=fig_pie)]),
    _card([html.H3("☁️ Word Cloud", style={"color": "#2c3e50"}),
           html.Img(src=f"data:image/png;base64,{img_base64}", style={"width": "100%", "maxWidth": "800px"})], center=True),
    _card([dcc.Graph(figure=fig_line)]),
    _card([dcc.Graph(figure=fig_hour)]),
    _card([dcc.Graph(figure=fig_bar)]),
    _card([dcc.Graph(figure=fig_topics)]),

    _card([
        html.H3(f"📰 Headlines (Total: {len(df_sorted)})", style={"color": "#2c3e50", "marginBottom": "20px"}),
        dash_table.DataTable(
            columns=[
                {"name": "Source", "id": "source"},
                {"name": "Title", "id": "title", "presentation": "markdown"},
                {"name": "Topics", "id": "topics_display"},
                {"name": "Sentiment", "id": "sentiment"},
                {"name": "Published At", "id": "publishedAt"}
            ],
            data=df_sorted[["source", "title", "topics_display", "sentiment", "publishedAt"]].to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px", "whiteSpace": "normal", "fontFamily": "Segoe UI"},
            style_header={"backgroundColor": "#e8eaf6", "fontWeight": "bold"},
            page_size=10,
            markdown_options={"link_target": "_blank"}
        )
    ])
])

if __name__ == "__main__":
    app.run(debug=True)
