import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
# Page config
st.set_page_config(
    page_title="IT Reviews Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        padding: 10px;
    }
    .metric-card {
        background-color: #f0f4ff;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    "<div class='main-header'>IT Customer Reviews — Analytics Dashboard</div>",
    unsafe_allow_html=True
)
st.markdown("---")

# Data load karo
@st.cache_data
def load_data():
    df = pd.read_excel("reviews_FINAL_complete.xlsx")
    return df

df = load_data()

# ─── SIDEBAR FILTERS ───────────────────────────────────────
st.sidebar.header("Filters")

# Sentiment filter
sentiment_options = ["All"] + list(df["Sentiment_Label"].unique())
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiment_options)

# Issue category filter
category_options = ["All"] + list(df["Issue_Category"].unique())
selected_category = st.sidebar.selectbox("Issue Category", category_options)

# Rating filter
rating_range = st.sidebar.slider(
    "Rating Range", 1, 5, (1, 5)
)

# Apply filters
filtered_df = df.copy()
if selected_sentiment != "All":
    filtered_df = filtered_df[
        filtered_df["Sentiment_Label"] == selected_sentiment
    ]
if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["Issue_Category"] == selected_category
    ]
filtered_df = filtered_df[
    filtered_df["Rating"].between(
        rating_range[0], rating_range[1]
    )
]

# ─── KPI CARDS ─────────────────────────────────────────────
st.subheader("Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

total = len(filtered_df)
pos   = len(filtered_df[filtered_df["Sentiment_Label"] == "Positive"])
neg   = len(filtered_df[filtered_df["Sentiment_Label"] == "Negative"])
neu   = len(filtered_df[filtered_df["Sentiment_Label"] == "Neutral"])
avg_css = round(filtered_df["CSS"].mean(), 3) if "CSS" in filtered_df.columns else "N/A"

col1.metric("Total Reviews",   total)
col2.metric("Positive",        pos,  f"{round(pos/total*100)}%")
col3.metric("Negative",        neg,  f"{round(neg/total*100)}%")
col4.metric("Neutral",         neu,  f"{round(neu/total*100)}%")
col5.metric("Avg CSS Score",   avg_css)

st.markdown("---")

# ─── TAB LAYOUT ────────────────────────────────────────────
# Aur replace karo with:
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Sentiment Analysis",
    "Issue Categories",
    "Word Clouds",
    "Model Comparison",
    "Raw Data",
    "Emoji Analysis"   # NEW TAB
])

# ── TAB 1: Sentiment Analysis ──
with tab1:
    st.subheader("Sentiment Distribution")
    col1, col2 = st.columns(2)

    with col1:
        sentiment_counts = filtered_df["Sentiment_Label"].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                "Positive": "#639922",
                "Negative": "#E24B4A",
                "Neutral":  "#888780"
            },
            title="Sentiment Split"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rating_counts = filtered_df["Rating"].value_counts().sort_index()
        fig2 = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            color=rating_counts.index,
            color_continuous_scale="Blues",
            title="Rating Distribution",
            labels={"x": "Rating", "y": "Count"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    # CSS over categories
    if "CSS" in filtered_df.columns:
        st.subheader("CSS Score by Issue Category")
        css_cat = filtered_df.groupby(
            "Issue_Category"
        )["CSS"].mean().round(3).sort_values()
        fig3 = px.bar(
            x=css_cat.values,
            y=css_cat.index,
            orientation="h",
            color=css_cat.values,
            color_continuous_scale="RdYlGn",
            title="Avg Customer Satisfaction Score",
            labels={"x": "CSS Score", "y": "Category"}
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── TAB 2: Issue Categories ──
with tab2:
    st.subheader("Issue Category Analysis")
    col1, col2 = st.columns(2)

    with col1:
        cat_counts = filtered_df["Issue_Category"].value_counts().head(10)
        fig = px.bar(
            x=cat_counts.values,
            y=cat_counts.index,
            orientation="h",
            color=cat_counts.values,
            color_continuous_scale="Blues",
            title="Top Issue Categories",
            labels={"x": "Count", "y": "Category"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pivot = pd.crosstab(
            filtered_df["Issue_Category"],
            filtered_df["Sentiment_Label"]
        )
        fig2 = px.imshow(
            pivot,
            color_continuous_scale="YlOrRd",
            title="Sentiment × Issue Category Heatmap",
            text_auto=True
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── TAB 3: Word Clouds ──
with tab3:
    st.subheader("Word Clouds by Sentiment")
    wc_option = st.radio(
        "Select sentiment:",
        ["Positive", "Negative", "Neutral"],
        horizontal=True
    )

    color_map = {
        "Positive": "Greens",
        "Negative": "Reds",
        "Neutral":  "Blues"
    }

    text = " ".join(
        filtered_df[
            filtered_df["Sentiment_Label"] == wc_option
        ]["processed_text"].dropna()
    )

    if text.strip():
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap=color_map[wc_option],
            max_words=100
        ).generate(text)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("Is filter ke liye koi data nahi!")

# ── TAB 4: Model Comparison ──
with tab4:
    st.subheader("NLP Model Accuracy Comparison")

    model_data = pd.DataFrame({
        "Model":    ["TextBlob", "VADER", "LSTM", "BiLSTM"],
        "Accuracy": [42.0, 40.0, 44.33, 67.0],
        "Type":     ["Rule-based", "Rule-based",
                     "Deep Learning", "Deep Learning"]
    })

    fig = px.bar(
        model_data,
        x="Model", y="Accuracy",
        color="Type",
        color_discrete_map={
            "Rule-based":    "#888780",
            "Deep Learning": "#378ADD"
        },
        title="Model Accuracy Comparison",
        text="Accuracy"
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key Findings")
    st.success(
        "BiLSTM achieved highest accuracy of 67% — "
        "significantly better than rule-based models (40-42%). "
        "Bidirectional context understanding helps capture "
        "complex IT review language."
    )
    st.info(
        "Rule-based models (TextBlob & VADER) struggled with "
        "professional IT language where sentiment is implied "
        "rather than explicit."
    )

# ── TAB 5: Raw Data ──
with tab5:
    st.subheader("Review Data Explorer")
    st.write(f"Showing {len(filtered_df)} reviews")

    cols_to_show = ["Review_Text", "Rating",
                "Sentiment_Label", "Issue_Category", "CSS"]
    available_cols = [c for c in cols_to_show if c in filtered_df.columns]
    st.dataframe(filtered_df[available_cols].reset_index(drop=True),use_container_width=True,height=400)

    # Download button
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_reviews.csv",
        mime="text/csv"
    )
    # Tab 6 content — end mein add karo:
with tab6:
    st.subheader("Emoji-Based Sentiment Analysis")

    # Load emoji data
    @st.cache_data
    def load_emoji_data():
        try:
            return pd.read_excel("emoji_sentiment_results.xlsx")
        except:
            return None

    df_em = load_emoji_data()

    if df_em is None:
        st.warning("emoji_sentiment_results.xlsx not found!")
    else:
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        total_em = len(df_em[df_em['emoji_count'] > 0])
        pos_em   = len(df_em[df_em['emoji_sentiment']=='Positive'])
        neg_em   = len(df_em[df_em['emoji_sentiment']=='Negative'])
        avg_css  = round(df_em['CSS_enhanced'].mean(), 3)

        col1.metric("Reviews with Emoji", total_em)
        col2.metric("Emoji Positive",     pos_em)
        col3.metric("Emoji Negative",     neg_em)
        col4.metric("Avg Enhanced CSS",   avg_css)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            # Emoji sentiment pie
            em_counts = df_em[
                df_em['emoji_sentiment'] != 'No Emoji'
            ]['emoji_sentiment'].value_counts()
            fig = px.pie(
                values=em_counts.values,
                names=em_counts.index,
                color=em_counts.index,
                color_discrete_map={
                    'Positive':'#639922',
                    'Negative':'#E24B4A',
                    'Neutral' :'#888780'
                },
                title="Emoji Sentiment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # CSS comparison
            has_e  = df_em[df_em['emoji_count']>0]['CSS_enhanced'].mean()
            no_e   = df_em[df_em['emoji_count']==0]['CSS_enhanced'].mean()
            fig2 = px.bar(
                x=['With Emoji','Without Emoji'],
                y=[round(has_e,3), round(no_e,3)],
                color=['With Emoji','Without Emoji'],
                color_discrete_map={
                    'With Emoji'   :'#378ADD',
                    'Without Emoji':'#888780'
                },
                title="Average CSS: Emoji vs No Emoji",
                labels={'x':'Group','y':'CSS Score'}
            )
            fig2.update_layout(yaxis_range=[0,1])
            st.plotly_chart(fig2, use_container_width=True)

        # App-wise CSS
        st.subheader("Enhanced CSS by App")
        css_app = df_em.groupby('app_name')['CSS_enhanced']\
                       .mean().round(3).sort_values()
        fig3 = px.bar(
            x=css_app.values, y=css_app.index,
            orientation='h',
            color=css_app.values,
            color_continuous_scale='RdYlGn',
            title="Customer Satisfaction Score by IT App",
            labels={'x':'CSS Score','y':'App'}
        )
        st.plotly_chart(fig3, use_container_width=True)
