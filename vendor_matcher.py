import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util

class VendorMatcher:
    def __init__(self, csv_path, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the matcher with dataset and embedding model.
        """
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer(model_name)
        self._preprocess_data()
        self._extract_feature_names()
        self._compute_feature_embeddings()

    def _preprocess_data(self):
        """
        Select relevant columns and generate combined text fields for embeddings.
        """
        relevant_columns = [
            'product_name', 'rating', 'description', 'reviews_count',
            'categories', 'overview', 'Features'
        ]
        self.df = self.df[relevant_columns].copy()

        # Fill missing values and combine fields
        text_field = ['description', 'categories', 'overview', 'Features']
        self.df[text_field] = self.df[text_field].fillna("").astype(str)

        # Text used for different embedding contexts
        category_field = ['description', 'categories', 'overview']
        self.df["category_search"] = self.df[category_field].agg(" ".join, axis=1)
        self.df["feature_search"] = self.df[text_field].agg(" ".join, axis=1)

    def _extract_feature_names(self):
        """
        Extract list of feature names from JSON string in 'Features' column.
        Also precompute feature name embeddings to avoid repetitive encoding.
        """
        def parse_names(feature_json):
            try:
                parsed = json.loads(feature_json)
                return [f["name"] for group in parsed for f in group.get("features", []) if "name" in f]
            except Exception:
                return []

        self.df["parsed_features"] = self.df["Features"].apply(parse_names)

        # Precompute embeddings for parsed feature names
        def embed_parsed(features):
            if not features:
                return None
            return self.model.encode(features, convert_to_tensor=True)

        self.df["parsed_feature_embeddings"] = self.df["parsed_features"].apply(embed_parsed)

    def _compute_feature_embeddings(self):
        """
        Compute sentence embeddings for category and feature search fields.
        """
        self.feature_embeddings_category = self.model.encode(
            self.df["category_search"].tolist(), convert_to_tensor=True
        )
        self.feature_embeddings_feature = self.model.encode(
            self.df["feature_search"].tolist(), convert_to_tensor=True
        )

    def calculate_category(self, df: pd.DataFrame, category: str, threshold: float = 0.3):
        """
        Filter software based on semantic similarity to category.
        """
        category_embedding = self.model.encode(category, convert_to_tensor=True)
        category_search_embeddings = self.model.encode(df["category_search"].tolist(), convert_to_tensor=True)
        similarity_scores = util.cos_sim(category_embedding, category_search_embeddings)[0].cpu().numpy()

        df = df.copy()
        df["category_similarity_score"] = similarity_scores
        return df[df["category_similarity_score"] > threshold].reset_index(drop=True)

    def calculate_capacities(self, capacities: list, threshold: float = 0.3,
                             feature_match_threshold: float = 0.5, hybrid_weight: float = 0.6):
        """
        Filter and score software based on feature relevance to user-requested capacities.
        Combines parsed feature name match and general text similarity.

        Improvements:
        - Uses precomputed embeddings for parsed features to speed up loop.
        """
        capacities_embeddings = self.model.encode(capacities, convert_to_tensor=True)

        # Check semantic similarity against parsed feature name embeddings
        def max_feature_name_similarity(embedding_tensor):
            if embedding_tensor is None:
                return 0.0
            sim_matrix = util.cos_sim(capacities_embeddings, embedding_tensor).cpu().numpy()
            return sim_matrix.max()

        parsed_sim_scores = self.df["parsed_feature_embeddings"].apply(max_feature_name_similarity)

        # Compute cosine similarities across full feature search field
        full_sim_matrix = util.cos_sim(capacities_embeddings, self.feature_embeddings_feature).cpu().numpy()
        max_sim = full_sim_matrix.max(axis=0)
        avg_sim = full_sim_matrix.mean(axis=0)
        hybrid_sim = hybrid_weight * max_sim + (1 - hybrid_weight) * avg_sim

        result_df = self.df.copy()
        result_df["parsed_feature_similarity"] = parsed_sim_scores
        result_df["capacity_max_sim"] = max_sim
        result_df["capacity_avg_sim"] = avg_sim
        result_df["capacity_hybrid_score"] = hybrid_sim

        # Grouping: primary = parsed match, secondary = fallback on full match
        primary_mask = result_df["parsed_feature_similarity"] >= feature_match_threshold
        secondary_mask = (~primary_mask) & (result_df["capacity_max_sim"] >= threshold)

        result_df["match_group"] = np.where(primary_mask, "primary",
                                            np.where(secondary_mask, "secondary", "none"))
        result_df = result_df[result_df["match_group"] != "none"].copy()
        result_df["group_rank"] = result_df["match_group"].map({"primary": 0, "secondary": 1})

        return result_df

    def calculate_bayesian_score(self, df: pd.DataFrame, percentile: float = 0.7):
        """
        Add Bayesian-adjusted rating score based on review count and global average.
        """
        df = df.copy()
        df["reviews_count"] = df["reviews_count"].fillna(0)

        C = df["rating"].mean()
        m = np.percentile(df["reviews_count"], percentile * 100)
        v = df["reviews_count"]
        R = df["rating"]

        df["bayesian_score"] = (v / (v + m)) * R + (m / (v + m)) * C
        return df

    def rank_final(self, df: pd.DataFrame, sim_weight: float = 0.6, bayes_weight: float = 0.4):
        """
        Combine semantic similarity and Bayesian trust score for final ranking.
        """
        df = df.copy()

        def normalize(col):
            min_val = col.min()
            max_val = col.max()
            if max_val == min_val:
                return np.ones_like(col)
            return (col - min_val) / (max_val - min_val)

        df["normalized_capacity_score"] = normalize(df["capacity_hybrid_score"])
        df["normalized_bayesian_score"] = normalize(df["bayesian_score"])

        df["final_score"] = sim_weight * df["normalized_capacity_score"] + bayes_weight * df["normalized_bayesian_score"]
        return df.sort_values(by=["group_rank", "final_score"], ascending=[True, False]).reset_index(drop=True)

    def recommend(self, capacities: list = None, category: str = None,
                  cap_threshold: float = 0.3, feature_match_threshold: float = 0.5,
                  sim_weight: float = 0.6, bayes_weight: float = 0.4,
                  bayes_percentile: float = 0.7, category_threshold: float = 0.3):
        """
        Main entry point to recommend vendors based on optional capacities and/or category.
        Handles four cases:
        - No input: returns by Bayesian score only
        - Only category: filters by category
        - Only capacities: filters by features
        - Both: filters by both and ranks accordingly
        """
        if not capacities and not category:
            # Case 1: No input
            df = self.calculate_bayesian_score(self.df, percentile=bayes_percentile)
            return df.sort_values(by="bayesian_score", ascending=False).reset_index(drop=True)

        if not capacities and category:
            # Case 2: Only category
            df = self.calculate_category(self.df, category, threshold=category_threshold)
            if df.empty:
                print("No software matched the given category.")
                return pd.DataFrame()
            df = self.calculate_bayesian_score(df, percentile=bayes_percentile)
            return df.sort_values(by="bayesian_score", ascending=False).reset_index(drop=True)

        if capacities and not category:
            # Case 3: Only capacities
            df = self.calculate_capacities(capacities, threshold=cap_threshold,
                                           feature_match_threshold=feature_match_threshold)
            if df.empty:
                print("No software matched the given capacities.")
                return pd.DataFrame()
            df = self.calculate_bayesian_score(df, percentile=bayes_percentile)
            return self.rank_final(df, sim_weight=sim_weight, bayes_weight=bayes_weight)

        # Case 4: Both capacities and category
        df = self.calculate_capacities(capacities, threshold=cap_threshold,
                                       feature_match_threshold=feature_match_threshold)
        if df.empty:
            print("No software matched the given capacities.")
            return pd.DataFrame()

        df = self.calculate_category(df, category, threshold=category_threshold)
        if df.empty:
            print("No software matched both category and capacities.")
            return pd.DataFrame()

        df = self.calculate_bayesian_score(df, percentile=bayes_percentile)
        return self.rank_final(df, sim_weight=sim_weight, bayes_weight=bayes_weight)




