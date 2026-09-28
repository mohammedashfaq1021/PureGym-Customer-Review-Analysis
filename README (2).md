# Gym Customer Review Analysis: Topic Modeling & Emotion Detection

**CAM C301 – Weeks 4 & 5 Topic Project**  
**Author:** Patel Mohammed

## Overview

This project analyses customer reviews of gym locations collected from **Google Reviews** and **Trustpilot**. The focus is on understanding common pain points in **negative reviews** (ratings < 3 stars) using classical NLP techniques, modern topic modeling (BERTopic), emotion classification, and generative AI.

The notebook covers the full pipeline from data loading and cleaning through exploratory analysis, topic discovery, location ranking, emotion analysis, and interactive topic visualisation with LDA.

## Datasets

| Source       | Original Size | Negative Reviews (rating < 3) |
|--------------|---------------|-------------------------------|
| Google       | 23,250        | 2,785                         |
| Trustpilot   | 16,673        | 3,543                         |

Key columns used:
- Review text
- Star rating
- Location / Club name

## Project Structure & Workflow

1. **Data Loading & Preparation**
   - Load Google and Trustpilot CSVs
   - Select relevant columns and standardise naming
   - Handle missing values

2. **Text Cleaning & Exploratory Word Analysis**
   - Lowercasing, punctuation removal, stopword filtering
   - Frequency analysis and bar charts of top words
   - WordClouds for all reviews

3. **Negative Review Analysis**
   - Filter reviews with rating < 3
   - Word frequency and WordClouds focused on negative feedback

4. **Topic Modelling with BERTopic**
   - Fit BERTopic on combined negative reviews from common locations
   - Visualisations: intertopic distance map, bar charts, heatmap
   - Key discovered themes include:
     - Air conditioning / temperature issues
     - Classes & booking problems
     - Cleanliness / smells / toilets
     - Access / pass codes
     - Equipment condition
     - Staff behaviour

5. **Worst Locations Analysis**
   - Rank locations by volume of negative reviews
   - Focused BERTopic on the top 30 locations
   - Additional WordCloud for high-complaint sites

6. **Emotion Analysis**
   - Transformer-based emotion classifier (`bhadresh-savani/bert-base-uncased-emotion`)
   - Dominant emotions in negative reviews: **anger**, sadness, joy (mixed), fear
   - Topic modelling specifically on “anger” reviews

7. **Generative AI Exploration (Falcon-7B-Instruct)**
   - Prompting a large language model to extract main topics from sample reviews
   - Lightweight BERTopic experiment on the generated topic outputs

8. **Classical Topic Modelling – Gensim LDA**
   - 10-topic LDA model on all negative reviews
   - Interactive visualisation with **pyLDAvis**

## Key Insights (from notebook)

- Negative reviews are dominated by **anger**.
- Recurring complaint themes:
  - Poor cleanliness (toilets, changing rooms, smells)
  - Broken or insufficient equipment
  - Air conditioning / temperature problems
  - Rude or unhelpful staff / managers
  - Access and membership issues (pass codes, bookings)
- Certain locations consistently attract higher volumes of negative feedback.

## Technologies Used

- **Python 3**
- **pandas** – data handling
- **NLTK** – tokenization & stopwords
- **WordCloud** + **matplotlib** – visualisation
- **BERTopic** (with UMAP + HDBSCAN + Sentence Transformers)
- **Hugging Face Transformers** – emotion classification & Falcon-7B
- **Gensim** + **pyLDAvis** – LDA topic modelling & interactive viz

## How to Run

This notebook was developed in **Google Colab** (with T4 GPU recommended for the larger models).

1. Upload the two CSV files:
   - `Google - Reviews - Sheet1.csv`
   - `TrustPilot - Reviews - Sheet1.csv`
2. Open the notebook in Colab.
3. Run cells sequentially. Several `!pip install` cells install the required packages.
4. Note: Falcon-7B and BERTopic can be memory-intensive; a GPU runtime is strongly advised.

### Main Dependencies

```bash
pandas
nltk
wordcloud
matplotlib
bertopic
transformers
accelerate
gensim
pyLDAvis
sentence-transformers
umap-learn
hdbscan
```

## Notes

- The notebook contains interactive visualisations (BERTopic plots and pyLDAvis) that render best in a Jupyter / Colab environment.
- Some steps sample data or use simplified parameters (especially the Falcon section) for demonstration purposes.
- Location names appear in both English and other languages (Danish/German terms appear in topic outputs), reflecting the multi-country nature of the review data.

## License

This project was created as coursework for CAM C301. Feel free to use the analysis approach and code structure for educational purposes.
