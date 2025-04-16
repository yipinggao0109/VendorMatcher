Overview:
    VendorMatcher is a semantic recommendation engine designed to identify and rank software products based on their relevance to user-defined categories and functional capacities. It uses sentence embeddings (via Sentence-BERT) to compare textual product data and employs Bayesian-adjusted ratings for trustworthy scoring.

1. Data Processing and Embedding Optimization:

    When the VendorMatcher object is created, it loads and processes key fields from the dataset including product name, rating, review count, description, categories, overview, and a JSON-formatted features column. Two combined text fields are constructed to support semantic search: one called "category_search" that merges description, categories, and overview, and another called "feature_search" that includes those fields plus the features. To save computation time during recommendations, all embeddings are computed once during initialization. This includes sentence embeddings for category_search, feature_search, and parsed feature names extracted from the JSON features column. Each list of parsed feature names is individually embedded and stored per row.

2. Matching Logic

    The system supports recommendations based on category, features, both, or neither. If no input is provided, products are ranked based on their Bayesian-adjusted rating. If only a category is given, the system filters for products with category similarity above a threshold. If only feature capacities are given, the system attempts to match those capacities against the parsed feature names first. If the highest similarity between user features and parsed features exceeds 0.5, the product is treated as a primary match. If that fails, the system compares the features against the broader feature_search text. A hybrid similarity score is computed using 60% of the maximum similarity and 40% of the average similarity across all features. If this hybrid score exceeds 0.3, the product is labeled a secondary match. If a category is also specified, the filtered results are further restricted to those with high category similarity (exceeds 0.3). Each product is assigned a match group: primary, secondary, or none. Only primary and secondary matches are considered in the final ranking.

3. Thresholds Choosing and Rating Score:

    Thresholds were selected using the threshold_selection.py program, which generates line plots showing the relationship between similarity scores and the number of matched products. These plots are stored in the threshold_selection_plots folder. Based on the elbow point of the graph, a threshold of 0.3 was chosen for both category and text-based feature matches. This value strikes a balance between relevance and recall. Setting the threshold too high would risk filtering out many potentially useful matches, especially when users submit narrow or highly specific queries. For example, many CRM tools include financial and accounting functions even if they are not explicitly marketed that way. By allowing a lower threshold, the system helps users discover these broader matches. A higher threshold of 0.5 is still used for parsed feature name comparisons, ensuring more confident and direct matches. For ranking, a Bayesian average is used to adjust raw ratings. This method helps reduce the influence of products with few reviews but high scores. The Bayesian formula blends the product's rating with the global average, weighted by the number of reviews and a minimum credibility threshold, defined as the 70th percentile of all review counts.

4. Final Ranking:

    The final ranking score is a weighted combination of semantic similarity and Bayesian trust. Each component is normalized between zero and one, and the final score is calculated as 60% semantic similarity and 40% Bayesian score. Results are then sorted by match group and final score, with primary matches ranked above secondary ones.

5. Edge Cases:

    Edge cases are handled through flexible thresholds and fallback logic. If no input is provided, the program returns vendors ranked by Bayesian score, helping users discover popular and well-reviewed software they may want to try. If a category or feature query yields no matches, it returns an empty DataFrame with a clear message instead of breaking. By using lower thresholds for broader matches and applying layered filtering, the system avoids failing on overly specific, vague, or incomplete queries while still prioritizing relevant results.

6. Challenges Encountered:

    One of the main challenges was balancing precision and recall when setting similarity thresholds. If the thresholds were too high, relevant results were often excluded; if too low, the matches became noisy and less useful. To address this, I developed a separate threshold analysis tool (threshold_selection.py) to visualize match counts across different cutoffs and identify optimal thresholds empirically. Another challenge was avoiding repeated computation of embeddings, especially when comparing large sets of feature names. To solve this, I precomputed and cached all embeddings during object initialization, significantly reducing runtime during queries. Parsing and embedding features from nested JSON structures also required careful handling to avoid silent failures or empty results. Lastly, ranking results fairly required adjusting raw ratings, which led to the integration of a Bayesian scoring method to account for products with few but possibly inflated reviews.

7. Potential Improvements:

    There are several potential improvements to enhance the system. Currently, the thresholds for similarity filtering are selected based on visual inspection of line plots. Incorporating human feedback or relevance-labeled data could help fine-tune these thresholds more systematically. Additionally, the final scoring weights between semantic similarity and Bayesian trust are based on intuition due to time constraints. In future iterations, I could design experiments or use grid search with evaluation metrics to determine the optimal weighting scheme. Other improvements may include introducing learning-to-rank models, allowing user-adjustable thresholds, or integrating metadata filters like deployment type or pricing.

Note: Parts of this documentation were refined with the help of AI (ChatGPT) for clarity and organization.




        





