import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_regional_growth(df: pd.DataFrame):
    """
    Creates and returns a bar chart of the average market growth rate by region.
    """
    regional_growth = df.groupby('Region')['Growth Rate (%)'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=regional_growth.index, y=regional_growth.values, palette='viridis', ax=ax)
    
    ax.set_title('Average Condom Market Growth Rate by Region (The "Party Regions" 🎉)', fontsize=16)
    ax.set_xlabel('Region', fontsize=12)
    ax.set_ylabel('Average Growth Rate (%)', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    return fig

def plot_brand_market_share(df: pd.DataFrame):
    """
    Creates and returns a bar chart of the average market share by brand.
    """
    brand_share = df.groupby('Brand Name')['Market Share (%)'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=brand_share.index, y=brand_share.values, palette='magma', ax=ax)
    
    ax.set_title('Average Market Share by Brand (Who’s the Real MVP? 🏆)', fontsize=16)
    ax.set_xlabel('Brand Name', fontsize=12)
    ax.set_ylabel('Average Market Share (%)', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    return fig

def plot_material_preference(df: pd.DataFrame):
    """
    Creates and returns a pie chart showing the preference for Latex vs. Non-latex.
    """
    material_preference = df['Material Type'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(material_preference, labels=material_preference.index, autopct='%1.1f%%', 
           startangle=140, colors=['#ff9999','#66b3ff'], textprops={'fontsize': 12})
    
    ax.set_title('Latex or Non-Latex? 🤔', fontsize=16)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    return fig

def plot_market_size_trend(df: pd.DataFrame):
    """
    Creates and returns a line chart of the total market size over the years.
    """
    market_size_trend = df.groupby('Year')['Market Size (USD Million)'].sum()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(market_size_trend.index, market_size_trend.values, marker='o', linestyle='-', color='purple')
    
    ax.set_title('Indian Condom Market Size Trend 📈', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Market Size (USD Million)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    return fig
