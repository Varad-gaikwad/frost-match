# 🍦 FrostMatch — AI Ice Cream Flavor Recommender

A neural network-based **collaborative filtering** recommender that predicts which ice cream flavors you'll love, built with a two-tower TensorFlow architecture and deployed as a polished, themeable Streamlit app.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Two--Tower%20NN-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Output / Demo
###https://frostmatch.streamlit.app/


<img width="1710" height="868" alt="image" src="https://github.com/user-attachments/assets/4089466b-41aa-421c-903c-346f60bce325" />
<img width="1710" height="868" alt="image" src="https://github.com/user-attachments/assets/dda1a5b1-7e37-4141-89b6-ddda13573451" />



---

## 🎯 Overview

Rate 2 (or more) ice cream flavors, and FrostMatch predicts your top recommendations from flavors you haven't tried — using a **two-tower neural network** (separate user and item embedding networks) trained on real flavor-rating data, the same architecture used in production-grade recommender systems (e.g., YouTube, Netflix-style models).

Unlike simple similarity-based recommenders, this model **learns** user and item embeddings from data via gradient descent, rather than relying on hand-crafted similarity rules.

---

## 🧠 How It Works

1. **Flavor features** (`ice_cream_flavors.csv`) — 65 flavors, each described by 12 attributes (sweetness, chocolate content, fruity, nutty, creaminess, intensity, etc.)
2. **User ratings** (`user_ratings.csv`) — real user → flavor → rating (1-5) data
3. **User vectors** — each user's taste profile is computed as a rating-weighted average of the flavors they've rated
4. **Two-Tower Neural Network** — a `user_NN` and `item_NN` (each: Dense(128) → Dense(64) → Dense(32)) independently embed users and flavors into a shared 32-dimensional space; a dot product between the two embeddings predicts the rating
5. **New user recommendation** — a new user rates a few flavors → their taste vector is built on the fly → the trained model scores every unrated flavor → top-N highest-scoring flavors are returned
6. **Custom flavors** (`custom_flavors.csv`) supported — add your own flavor + feature profile without retraining

---

## 📁 Repository Structure

```
frostmatch-recommender/
│
├── recommeder.py             # Model architecture, training, save/load
├── app.py                    # Streamlit app (FrostMatch UI, theming)
├── ice_cream_flavors.csv     # Flavor feature dataset
├── user_ratings.csv          # User rating dataset
├── custom_flavors.csv        # User-added custom flavors
├── ice_cream_model.keras     # Saved trained model
├── item_features_indexed.pkl # Saved flavor feature lookup
└── README.md
```
---

## 🔍 Key ML Concepts Demonstrated

- **Collaborative filtering** via learned embeddings (not hand-coded similarity)
- **Two-tower neural architecture** with a `Dot` layer combining user/item embeddings
- **Early stopping** (`patience=10`, `restore_best_weights=True`) to prevent overfitting
- **Cold-start handling** — new users get a taste vector built instantly from just 2+ ratings, no retraining needed
- **Output rescaling** — raw dot-product scores are rescaled to a 1-5 range for interpretability

---

## 🛠️ Tech Stack

TensorFlow/Keras · Pandas · NumPy · Scikit-learn · Streamlit · Joblib

---

## 👤 Author

**Varad Tushar Gaikwad** — [GitHub](https://github.com/Varad-gaikwad)

## 📄 License

MIT License — see [LICENSE](LICENSE).
