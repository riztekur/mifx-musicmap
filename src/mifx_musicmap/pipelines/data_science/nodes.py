from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pandas as pd
import scorecardpy as sc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def user_clustering(df):
    df = df[['songsListened_monthly', 'tenure', 'subscription_spending_$']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df['cluster'] = labels
    return df

def train_model(train):
    logreg = LogisticRegression(random_state=42)
    X = train.drop(columns=["adopter"])
    y = train["adopter"]
    logreg.fit(X, y)
    return logreg

def train_tree_model(df):
    tree=DecisionTreeClassifier(max_depth=5, random_state=42)
    X = df.drop(columns=["adopter"])
    y = df["adopter"]
    tree.fit(X, y)

    accuracy = tree.score(X, y)

    try:
        y_prob = tree.predict_proba(X)[:, 1]  # Probability estimates for the positive class
        roc_auc = roc_auc_score(y, y_prob)
    except:
        roc_auc = None

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_tree(tree, filled=True, feature_names=None, class_names=True, ax=ax)

    score_text = f"Accuracy: {accuracy:.2f}"
    if roc_auc is not None:
        score_text += f"\nROC AUC: {roc_auc:.2f}"

    ax.text(0.5, -0.1, score_text, 
            ha="center", va="center", fontsize=12, transform=ax.transAxes, 
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))

    ax.set_title("Decision Tree Visualization")
    
    plt.tight_layout()
    return tree, fig

def evaluate_model(regressor, bins, train, test):
    X_train = train.drop(columns=["adopter"])
    y_train = train["adopter"]
    X_test = test.drop(columns=["adopter"])
    y_test = test["adopter"]

    train_pred = regressor.predict_proba(X_train)[:,1]
    test_pred = regressor.predict_proba(X_test)[:,1]

    train_pred_binary = (train_pred >= 0.5).astype(int)
    test_pred_binary = (test_pred >= 0.5).astype(int)

    train_perf = sc.perf_eva(y_train.squeeze(), train_pred, title = "train", plot_type=["ks", "roc"])
    test_perf = sc.perf_eva(y_test.squeeze(), test_pred, title = "test", plot_type=["ks", "roc"])

    performance_summary = {
        'KS': {
            'train': train_perf['KS'],
            'test': test_perf['KS']
        },
        'AUC': {
            'train': train_perf['AUC'],
            'test': test_perf['AUC']
        },
        'Gini': {
            'train': train_perf['Gini'],
            'test': test_perf['Gini']
        },
        'Accuracy': {
            'train': accuracy_score(y_train, train_pred_binary),
            'test': accuracy_score(y_test, test_pred_binary)
        },
        'Recall': {
            'train': recall_score(y_train, train_pred_binary),
            'test': recall_score(y_test, test_pred_binary)
        },
        'Precision': {
            'train': precision_score(y_train, train_pred_binary),
            'test': precision_score(y_test, test_pred_binary)
        }
    }

    card = sc.scorecard(bins, regressor, X_train.columns)
    card_df = pd.concat(card.values(), ignore_index=True)

    return performance_summary, train_perf['pic'], test_perf['pic'], card_df