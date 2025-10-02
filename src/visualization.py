import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_market_size_trend(df: pd.DataFrame):
    """
    Creates and returns a line chart of the total market size over the years.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    yearly_market_size = df.groupby('Year')['Market Size (USD Million)'].sum().reset_index()
    sns.lineplot(x='Year', y='Market Size (USD Million)', data=yearly_market_size, marker='o', color='darkblue', ax=ax)
    
    ax.set_title('Total Market Size (USD Million) Trend (2018-2030)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Market Size (USD Million)')
    ax.grid(True, which='both', linestyle='--')
    
    return fig

def plot_regional_growth(df: pd.DataFrame):
    """
    Creates and returns a horizontal bar chart of average market growth by region.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    region_growth = df.groupby('Region')['Growth Rate (%)'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(x='Growth Rate (%)', y='Region', data=region_growth, orient='h', palette='viridis', ax=ax)
    
    ax.set_title('Average Market Growth Rate (%) by Region')
    ax.set_xlabel('Average Growth Rate (%)')
    ax.set_ylabel('Region')
    
    return fig

def plot_brand_market_share(df: pd.DataFrame):
    """
    Creates and returns a box plot of market share distribution by brand.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(x='Market Share (%)', y='Brand Name', data=df, palette='magma', ax=ax)
    
    ax.set_title('Market Share (%) Distribution for Each Brand')
    ax.set_xlabel('Market Share (%)')
    ax.set_ylabel('Brand Name')
    
    return fig

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Creates and returns a correlation heatmap of all numerical features.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    numeric_cols = df.select_dtypes(include=np.number)
    corr_matrix = numeric_cols.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5, ax=ax)
    
    ax.set_title('Correlation Matrix of Numerical Features')
    
    return fig

def plot_brand_region_heatmap(df: pd.DataFrame):
    """
    Creates a pivot heatmap of average market share by brand and region.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_table = df.pivot_table(values='Market Share (%)', index='Brand Name', columns='Region', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=.5, ax=ax)
    
    ax.set_title('Average Market Share (%) by Brand and Region')
    ax.set_xlabel('Region')
    ax.set_ylabel('Brand Name')
    
    return fig

def plot_top_company_activity(df: pd.DataFrame):
    """
    Creates a horizontal bar chart of the Top 10 most active companies.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    top_10_companies = df['Company Involved'].value_counts().nlargest(10).sort_values(ascending=True)
    top_10_companies.plot(kind='barh', color=sns.color_palette('terrain', 10), ax=ax)
    
    ax.set_title('Top 10 Most Active Companies (by Number of Events)')
    ax.set_xlabel('Number of Events / Activities')
    ax.set_ylabel('Company')
    
    return fig
