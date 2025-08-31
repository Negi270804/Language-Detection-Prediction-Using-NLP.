# Language-Detection-Prediction-Using-NLP
---

# **Multilingual Language Detection**

A Python-based language detection system that can identify 22 different languages, including English, Hindi, Portuguese, Latin, Chinese, Tamil, and more. The project supports both **short and long text inputs** and can be integrated with **Streamlit** for an interactive web application.

---

## **Features**

* Detects **22 different languages** from text input.
* Works with **short sentences**, **long paragraphs**, and **different scripts** (Latin, Devanagari, Chinese, Tamil, etc.).
* Uses **TfidfVectorizer + Multinomial Naive Bayes** or optionally **`langdetect`/`fasttext`** for more robust short-text detection.
* Can be deployed as a **Streamlit app** for an interactive interface.
* High accuracy (95%+) on the test dataset.

---

## **Languages Supported**

* English
* Hindi
* Portuguese
* Latin
* French
* Chinese
* Japanese
* Tamil
* Thai
* Dutch
* Turkish
* Latin
* Urdu
* Indonesian
* Portuguese
* Spanish
* Pushto
* Persian
* Romanian
* Russian
* Arabic


---

## **Project Structure**

```
language-detection/
│
├── language.csv           # Dataset containing text and language labels
├── app.py                 # Streamlit web application
├── train_model.ipynb      # Jupyter notebook for training the model
├── vectorizer.pkl         # Saved TF-IDF vectorizer
├── model.pkl              # Saved MultinomialNB model
└── README.md              # Project documentation
```

---

## **Setup Instructions**

1. **Clone the repository**

```bash
git clone <repository-url>
cd language-detection
```

2. **Install required packages**

```bash
pip install -r requirements.txt
# Or manually:
pip install pandas numpy scikit-learn streamlit langdetect
```

3. **Train the model (optional)**

   * Open `train_model.ipynb` in Jupyter Notebook.
   * Run all cells to train the model on `language.csv`.
   * This will generate `vectorizer.pkl` and `model.pkl`.

4. **Run the Streamlit app**

```bash
streamlit run app.py
```

* A browser window will open with a text input box.
* Enter any text in any of the 22 languages and click **Detect Language**.
* The predicted language will appear instantly.

---

## **Usage Example (Python/Jupyter)**

```python
from langdetect import detect

text = "I am Nikhil Negi"
language = detect(text)
print("Predicted Language:", language)
```

**Output:**

```
Predicted Language: en
```

---

## **Streamlit Example**

1. Open `app.py`.
2. Type or paste your text into the text area.
3. Click **Detect Language**.
4. The app will display the predicted language.

---

## **Optional Improvements**

* Use **FastText pre-trained language detection model** for higher accuracy.
* Combine **word-level TF-IDF + char-level TF-IDF** for more robust predictions.
* Add **example buttons** for users to test all 22 languages.
* Deploy the Streamlit app online using **Streamlit Cloud** or **Heroku**.

---

## **References**

* [Scikit-learn TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
* [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
* [Langdetect Documentation](https://pypi.org/project/langdetect/)
* [FastText Language Identification](https://fasttext.cc/docs/en/language-identification.html)

---
## **Contact**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Nikhil%20Negi-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nikhil-negi-0bb166328)  
[![Email](https://img.shields.io/badge/Email-neginikhil424@gmail.com-red?logo=gmail&logoColor=white)](mailto:neginikhil424@gmail.com)
