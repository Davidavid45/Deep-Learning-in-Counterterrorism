"""
Text Processing for Summary Field

Processes attack summary text for optional text-based features
or analysis. Includes cleaning, tokenization, and vectorization.
"""

import pandas as pd
import numpy as np
import re
import sys
import os
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_config, load_processed_data, save_processed_data


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    if pd.isna(text) or text == 'Unknown':
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract key terms from text (simple frequency-based).
    
    Args:
        text: Input text
        top_n: Number of top keywords to extract
        
    Returns:
        List of keywords
    """
    if not text:
        return []
    
    # Simple word frequency (could be improved with TF-IDF)
    words = text.split()
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                  'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                  'were', 'be', 'been', 'being', 'have', 'has', 'had'}
    
    words = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Get unique words (maintaining order)
    seen = set()
    keywords = []
    for word in words:
        if word not in seen:
            keywords.append(word)
            seen.add(word)
            if len(keywords) >= top_n:
                break
    
    return keywords


def calculate_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic text statistics.
    
    Args:
        df: DataFrame with 'summary' column
        
    Returns:
        DataFrame with text statistics added
    """
    print("\nCalculating text statistics...")
    
    if 'summary' not in df.columns:
        print("  'summary' column not found")
        return df
    
    # Clean text
    df['summary_clean'] = df['summary'].apply(clean_text)
    
    # Word count
    df['summary_word_count'] = df['summary_clean'].apply(lambda x: len(x.split()) if x else 0)
    
    # Character count
    df['summary_char_count'] = df['summary_clean'].apply(len)
    
    # Has summary indicator
    df['has_summary'] = (df['summary_word_count'] > 0).astype(int)
    
    print(f"  Records with summary: {df['has_summary'].sum():,} ({df['has_summary'].mean():.1%})")
    print(f"  Average word count: {df['summary_word_count'].mean():.1f}")
    
    return df


def extract_summary_keywords(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Extract keywords from summaries.
    
    Args:
        df: DataFrame with 'summary_clean' column
        top_n: Number of keywords to extract
        
    Returns:
        DataFrame with keywords column
    """
    print(f"\nExtracting top {top_n} keywords from summaries...")
    
    if 'summary_clean' not in df.columns:
        print("  'summary_clean' column not found")
        return df
    
    df['keywords'] = df['summary_clean'].apply(lambda x: extract_keywords(x, top_n))
    df['keywords_str'] = df['keywords'].apply(lambda x: ', '.join(x) if x else '')
    
    # Count keyword occurrences
    all_keywords = [kw for keywords in df['keywords'] for kw in keywords]
    print(f"  Total unique keywords: {len(set(all_keywords)):,}")
    
    return df


def process_text_data(config: dict) -> pd.DataFrame:
    """
    Main text processing pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with processed text features
    """
    print("="*60)
    print("PROCESSING TEXT DATA")
    print("="*60)
    
    # Load cleaned data
    df = load_processed_data("01_cleaned_data.csv", config)
    print(f"Input shape: {df.shape}")
    
    # Process text
    df = calculate_text_stats(df)
    df = extract_summary_keywords(df)
    
    # Drop intermediate columns to save space
    df = df.drop(['summary_clean', 'keywords'], axis=1, errors='ignore')
    
    print(f"\nFinal shape: {df.shape}")
    
    print("\n" + "="*60)
    print("TEXT PROCESSING COMPLETE")
    print("="*60)
    
    return df


if __name__ == "__main__":
    config = load_config()
    df_text = process_text_data(config)
    
    # Save processed data
    save_processed_data(df_text, "07_text_processed.csv", config)
    
    print("\nText features added:")
    text_cols = ['summary_word_count', 'summary_char_count', 'has_summary', 'keywords_str']
    for col in text_cols:
        if col in df_text.columns:
            print(f"  - {col}")
    
    # Show examples with summaries
    with_summary = df_text[df_text['has_summary'] == 1].head(3)
    if len(with_summary) > 0:
        print("\nExample summaries:")
        for idx, row in with_summary.iterrows():
            print(f"\n  Keywords: {row['keywords_str']}")
            print(f"  Words: {row['summary_word_count']}")
