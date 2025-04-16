from flask import Flask, request, jsonify
from vendor_matcher import VendorMatcher

# Initialize Flask app
app = Flask(__name__)

# Initialize VendorMatcher with the dataset path
matcher = VendorMatcher(csv_path="data/G2 software - CRM Category Product Overviews.csv")

@app.route('/vendor_qualification', methods=['POST'])
def vendor_qualification():
    """
    API endpoint to recommend software vendors based on category and/or capabilities.
    Accepts POST requests with JSON input like:
    {
        "software_category": "CRM",
        "capabilities": ["Budgeting", "Customization"]
    }
    Returns top 10 matching products with product name and rating.
    """
    try:
        # Parse input JSON
        data = request.json
        category = data.get("software_category", None)
        capabilities = data.get("capabilities", None)

        # Get recommendation results from VendorMatcher
        result_df = matcher.recommend(
            capacities=capabilities,
            category=category
        )

        # Handle case where no vendors are matched
        if result_df.empty:
            return jsonify({"message": "No suitable vendors found."}), 200

        # Extract top 10 results and return as JSON
        top_10 = result_df.head(10).reset_index(drop=True)
        top_10["rank"] = top_10.index + 1
        output = top_10[["rank", "product_name", "rating", "reviews_count"]].to_dict(orient="records")
        return jsonify(output)

    except Exception as e:
        # Return error details if something goes wrong
        return jsonify({"error": str(e)}), 500

# Run the Flask app locally on port 5000
if __name__ == '__main__':
    app.run(debug=True, port=5000)