# IT Customer Reviews — NLP Analytics Dashboard

## Project Overview
This research project analyzes IT company customer reviews using NLP techniques
to extract sentiment insights and present them through an interactive dashboard.

## Research Objectives
- **Objective 1**: Identify key insights and metrics from IT customer reviews
- **Objective 2**: Design analytical dashboards to visualize customer feedback

## Dataset
- 1,500 raw IT customer reviews (final_combined_1500_raw.xlsx)
- 1,300 advanced annotated reviews (advanced_IT_reviews_dataset.xlsx)

## NLP Pipeline

| Phase | Task |
|-------|------|
| Phase 1 | Data Loading & Inspection |
| Phase 2 | Text Preprocessing (Tokenization, Lemmatization) |
| Phase 3 | Sentiment Analysis (TextBlob, VADER, CSS) |
| Phase 4 | Clustering (TF-IDF + K-Means) |
| Phase 5 | Visualization & Word Clouds |
| Phase 6 | Deep Learning (LSTM & BiLSTM) |

## Model Results

| Model | Accuracy | Type |
|-------|----------|------|
| TextBlob | 42.0% | Rule-based |
| VADER | 40.0% | Rule-based |
| LSTM | 44.33% | Deep Learning |
| **BiLSTM** | **67.0%** | **Deep Learning** |

## Key Findings
- BiLSTM achieved highest accuracy of **67%**
- Performance is the top pain point (402 reviews)
- Speed and Interface are top strength areas
- 52.2% reviews show declining trend

## Tech Stack
- Python, Pandas, NLTK, TextBlob, VADER
- Scikit-learn (TF-IDF, K-Means)
- TensorFlow/Keras (LSTM, BiLSTM)
- Streamlit (Dashboard)
- Plotly, Matplotlib, Seaborn, WordCloud

## How to Run Dashboard
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Project Structure
IT-Reviews-NLP-Analysis/
├── dashboard.py                    # Streamlit dashboard
├── requirements.txt                # Dependencies
├── reviews_FINAL_complete.xlsx     # Final processed data
├── images/
│   ├── model_comparison.png
│   ├── wordcloud_positive.png
│   ├── wordcloud_negative.png
│   ├── bilstm_confusion_matrix.png
│   └── training_history.png
└── README.md
