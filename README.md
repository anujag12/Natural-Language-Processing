# Customer Feedback Intelligence

An end to end NLP pipeline that finds complaints in 22,641 real customer reviews of women's clothing. It cleans the text, trains a classical baseline and a fine-tuned DistilBERT, builds embeddings for similarity search, and checks the results for fairness between departments.

The best classical model reached 0.891 accuracy and 0.837 macro F1 on 4,529 held-out reviews. On a smaller shared split, fine-tuned DistilBERT averaged 0.855 macro F1 over four seeds, against 0.805 for the baseline.

## Dataset

The project uses the Women's Clothing E-Commerce Reviews dataset from the Hugging Face Hub.

- Link: https://huggingface.co/datasets/saattrupdan/womens-clothing-ecommerce-reviews
- It is a processed copy of the Kaggle dataset by Nicapotato: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

Each row has the review text, a rating, a recommended flag, the age and three product columns. The notebook loads it from the Hub, so you do not need to download anything.

The text is real and unsanitised. I did not fix any spelling or punctuation. A review with `recommended_ind = 0` is treated as a complaint. 81.9% of the reviews are recommended and 18.1% are not.

## What is in the notebook

`NLP.ipynb` has five parts.

1. **Preprocessing and exploration.** NLTK cleaning with the negation words kept, spaCy lemmas and named entities, word and word pair counts, and an NLTK against spaCy comparison.
2. **Classical baseline.** TF-IDF with Logistic Regression and Complement Naive Bayes. It also has a cut-off test, 5-fold cross-validation and a test of NER count features.
3. **Embeddings.** Averaged GloVe vectors, Sentence-Transformer vectors, PCA and t-SNE plots, and a retrieval function that finds similar reviews.
4. **Transformer.** `distilbert-base-uncased` fine-tuned with the Hugging Face Trainer, a four-seed check and a comparison with Logistic Regression on the same data.
5. **Responsible NLP.** Per-department accuracy and complaint recall, a decision table, and a short review of risks.

The last cells build a small Gradio dashboard. You paste a raw review and it shows the Logistic Regression prediction, the DistilBERT prediction and the three most similar reviews.

## Results

| Model | Test reviews | Accuracy | Macro F1 |
|---|---|---|---|
| Logistic Regression, full training set | 4,529 | 0.891 | 0.837 |
| Complement Naive Bayes, full training set | 4,529 | 0.864 | 0.807 |
| Logistic Regression, same 2,800 training reviews as DistilBERT | 800 | 0.868 | 0.805 |
| DistilBERT, seed 42 | 800 | 0.889 | 0.825 |
| DistilBERT, mean of 4 seeds | 800 | 0.912 | 0.855 |

Some other findings:

- Without class weights, Logistic Regression found only 19% of the complaints. DistilBERT found 78%.
- DistilBERT trained in 60 seconds on a T4 GPU. Logistic Regression trained in 0.1 seconds on a CPU.
- Averaged GloVe gave a mean cosine similarity of 0.925 between random review pairs. MiniLM gave 0.366. So averaged GloVe separates reviews poorly.
- Complaint recall was 0.774 for Bottoms and 0.895 for Dresses. One overall score hid the weaker group.

## How to run

1. Open `NLP.ipynb` in Google Colab.
2. Set the runtime to a GPU. I used a Tesla T4.
3. Run all cells from the top. The first cell installs the libraries and the later ones download the data and the pretrained models.

Task 4 is slow on a CPU. The fine-tuned model is saved to the folder `best_distilbert_feedback`, and the Gradio dashboard loads it from there.

The random seed is 42 everywhere. Every library version is printed by the notebook. The main ones are:

| Library | Version |
|---|---|
| python | 3.13.15 |
| scikit-learn | 1.6.1 |
| nltk | 3.9.1 |
| spacy | 3.8.16 |
| gensim | 4.4.0 |
| sentence-transformers | 5.7.0 |
| transformers | 5.17.0 |
| torch | 2.11.0+cu130 |
| gradio | 6.27.0 |

## Limitations

- All results come from one train and test split. Cross-validation was run only for the classical models.
- The DistilBERT comparison uses a sample of 4,000 reviews, so small differences between models are only hints.
- Not recommended is only a proxy for a complaint. Some labels look wrong when you read the reviews.
- The dataset copy looks lowercased, and it has 845 fewer rows than the 23,486 described for the Kaggle original. I did not check which rows are missing.
- The data covers one retailer, women's clothing and English only.
- I did not test stop word removal on the classifier, and I did not test retrieval-augmented generation.
- The Gradio dashboard loads the saved seed 42 model, which was the weakest of the four seeds.

## Libraries and models

- NLTK, spaCy `en_core_web_sm`, scikit-learn, gensim, Hugging Face Transformers and Datasets, Sentence-Transformers, Gradio.
- GloVe `glove-wiki-gigaword-50`, `all-MiniLM-L6-v2` and `distilbert-base-uncased`.
