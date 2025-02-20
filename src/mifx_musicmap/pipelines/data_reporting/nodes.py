import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def plot_univariate(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(len(num_cols), 1, figsize=(8, 4 * len(num_cols)))

    if len(num_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, num_cols):
        sns.histplot(df[col], kde=True, bins=50, color="skyblue", ax=ax)
        ax.axvline(df[col].median(), color="red", linestyle="dashed", linewidth=2, label=f"Median: {df[col].median():.2f}")
        ax.axvline(df[col].mean(), color="green", linestyle="dashed", linewidth=2, label=f"Mean: {df[col].mean():.2f}")
        ax.set_title(f"Distribution of {col}")
        ax.legend()

    plt.tight_layout()
    return fig

def plot_correlation(df):
    correlation_matrix = df.corr(method='spearman',numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    return fig

def plot_tenure_correlation(df):
    """
    Create a bar chart of correlation values with 'tenure' (excluding itself) and return the figure.
    """
    # Compute correlation with 'tenure' and drop itself
    correlation_values = df.corr(method='spearman', numeric_only=True)['tenure'].drop('tenure').sort_values(ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    correlation_values.plot(kind='bar', color="skyblue", ax=ax)
    
    # Formatting
    ax.set_title("Correlation with Tenure (Excluding Itself)")
    ax.set_ylabel("Correlation Coefficient")
    ax.axhline(0, color="black", linewidth=1)  # Add a horizontal line at zero
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x labels for readability

    return fig

def plot_adopter_vs_non_adopter(df):
    """
    Create histograms for all numerical variables, comparing adopters vs. non-adopters.
    Returns a Matplotlib figure.
    """
    # Select numerical columns (excluding 'adopter')
    num_cols = df.select_dtypes(include=['number']).columns.drop('adopter', errors='ignore')

    # Create figure
    fig, axes = plt.subplots(len(num_cols), 1, figsize=(8, 5 * len(num_cols)))  
    if len(num_cols) == 1:  # Handle single-column case
        axes = [axes]

    for ax, col in zip(axes, num_cols):
        # Compute medians
        medians = df.groupby("adopter")[col].median()
        median_adopter = medians.get(1, 0)  # Handle missing categories
        median_not_adopter = medians.get(0, 0)

        # Get data
        adopter_hist = df[df["adopter"] == 1][col]
        not_adopter_hist = df[df["adopter"] == 0][col]

        # Plot histograms
        ax.hist(not_adopter_hist, bins=50, alpha=0.6, label='Not Adopter', color='red', edgecolor='black', density=True)
        ax.hist(adopter_hist, bins=50, alpha=0.6, label='Adopter', color='blue', edgecolor='black', density=True)

        # Add median lines
        ax.axvline(median_adopter, color="blue", linestyle="dashed", label=f"Adopter Median: {median_adopter:.2f}")
        ax.axvline(median_not_adopter, color="red", linestyle="dashed", label=f"Not Adopter Median: {median_not_adopter:.2f}")

        # Formatting
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {col} by User Type')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def mann_whitney_test(df):
    """
    Perform Mann-Whitney U test for all numerical columns, comparing adopters vs. non-adopters.
    Returns a DataFrame with Variable, P-value, Statistic, and Decision.
    """
    # Select numerical columns (excluding 'adopter' itself)
    num_cols = df.select_dtypes(include=['number']).columns.drop('adopter', errors='ignore')

    test_results = []

    for col in num_cols:
        # Separate groups
        adopter_values = df[df["adopter"] == 1][col].dropna()
        non_adopter_values = df[df["adopter"] == 0][col].dropna()

        # Perform Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(adopter_values, non_adopter_values, alternative="two-sided")

        # Decision: Reject or Fail to Reject
        decision = "Reject H₀" if p_value < 0.05 else "Fail to Reject H₀"

        # Append results
        test_results.append({"Variable": col, "P-value": p_value, "Statistic": stat, "Decision": decision})

    # Convert to DataFrame
    test_results = pd.DataFrame(test_results)

    return test_results

def chi_square_test(df):
    """
    Perform Chi-Square test for all categorical columns to compare adopters vs. non-adopters.
    Returns a DataFrame with Variable, P-value, Statistic, and Decision.
    """
    # Select categorical columns
    cat_cols = df.select_dtypes(exclude=['number']).columns.drop(['adopter','net_user'], errors='ignore')

    results = []

    for col in cat_cols:
        # Create contingency table
        contingency_table = pd.crosstab(df[col], df["adopter"])

        # Skip if only one category (no variation)
        if contingency_table.shape[0] < 2:
            continue  

        # Perform Chi-Square test
        stat, p_value, _, _ = stats.chi2_contingency(contingency_table)

        # Decision: Reject or Fail to Reject H₀
        decision = "Reject H₀" if p_value < 0.05 else "Fail to Reject H₀"

        # Append results
        results.append({"Variable": col, "P-value": p_value, "Statistic": stat, "Decision": decision})

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def plot_songs_vs_tenure(df):
    """
    Generate and return a scatter plot of Songs Listened vs Tenure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="tenure", y="songsListened_monthly", alpha=0.5, ax=ax)
    
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Songs Listened Monthly")
    ax.set_title("Scatter Plot of Songs Listened vs Tenure")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    return fig