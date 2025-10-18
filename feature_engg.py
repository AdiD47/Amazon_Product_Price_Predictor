import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from uitil import (
    extract_text_features,
    extract_numeric_features,
    extract_category_keywords,
    preprocess_unit
)


class FeatureEngineer:
    """
    Feature engineering pipeline with:
    - TF-IDF
    - Numeric & categorical features
    - Optional text embeddings
    """

    def __init__(self, max_tfidf_features: int = 100, use_embeddings: bool = True):
        self.max_tfidf_features = max_tfidf_features
        self.use_embeddings = use_embeddings
        self.tfidf_vectorizer = None
        self.unit_encoder = None
        self.feature_names = None
        if self.use_embeddings:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit TF-IDF vectorizer and unit encoder on training data.
        """
        df = extract_text_features(df.copy())

        # Fit TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.tfidf_vectorizer.fit(df['full_text'].fillna(''))

        # Fit unit encoder
        df['ipq_unit_clean'] = df['ipq_unit'].apply(preprocess_unit)
        self.unit_encoder = LabelEncoder()
        self.unit_encoder.fit(df['ipq_unit_clean'].fillna('unknown'))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted feature engineering pipeline.
        """
        df = extract_text_features(df.copy())

        # TF-IDF features
        tfidf_matrix = self.tfidf_vectorizer.transform(df['full_text'].fillna(''))
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
            index=df.index
        )

        # Numeric & category features
        numeric_features = df['full_text'].apply(extract_numeric_features)
        numeric_df = pd.DataFrame(numeric_features.tolist(), index=df.index)

        category_features = df['full_text'].apply(extract_category_keywords)
        category_df = pd.DataFrame(category_features.tolist(), index=df.index)

        # Basic features
        basic_features = pd.DataFrame({
            'text_length': df['text_length'],
            'word_count': df['word_count'],
            'item_name_length': df['item_name_length'],
            'description_length': df['description_length'],
            'bullet_points_length': df['bullet_points_length'],
            'ipq_value': np.log1p(df['ipq_value'].fillna(0)),  # log-transform numeric column
            'has_ipq_value': (~df['ipq_value'].isna()).astype(int)
        }, index=df.index)

        # Unit encoding
        df['ipq_unit_clean'] = df['ipq_unit'].apply(preprocess_unit)
        unit_encoded = self.unit_encoder.transform(df['ipq_unit_clean'].fillna('unknown'))
        unit_df = pd.DataFrame({'unit_encoded': unit_encoded}, index=df.index)

        # Image presence
        df['has_image'] = (~df['image_link'].isna() & (df['image_link'] != '')).astype(int)
        image_df = pd.DataFrame({'has_image': df['has_image']}, index=df.index)

        # Text embeddings
        if self.use_embeddings:
            text_cols = ['item_name', 'description']
            combined_text = df[text_cols].fillna('').agg(' '.join, axis=1)
            embeddings = self.embedding_model.encode(combined_text.tolist(), show_progress_bar=False)
            embedding_df = pd.DataFrame(embeddings, index=df.index)
        else:
            embedding_df = pd.DataFrame(index=df.index)

        # Combine all features
        features_df = pd.concat([
            basic_features,
            numeric_df,
            category_df,
            unit_df,
            image_df,
            tfidf_df,
            embedding_df
        ], axis=1)

        self.feature_names = features_df.columns.tolist()
        return features_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        """
        self.fit(df)
        return self.transform(df)


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame, use_embeddings: bool = True) -> tuple:
    """
    Engineer features for both training and test datasets with optional embeddings.
    Applies log transformation to ipq_value and prepares X_train/X_test.
    """
    feature_engineer = FeatureEngineer(max_tfidf_features=100, use_embeddings=use_embeddings)

    X_train = feature_engineer.fit_transform(train_df)
    X_test = feature_engineer.transform(test_df)

    # Log-transform target price safely
    y_train = np.log1p(train_df['price'].values) if 'price' in train_df.columns else None

    return X_train, X_test, y_train, feature_engineer
