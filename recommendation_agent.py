import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

def fetch_embeddings(table_name):
    conn = sqlite3.connect("ecommerce.db")
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if "embeddings" in df.columns:
        df["embeddings"] = df["embeddings"].apply(
            lambda x: np.array(list(map(float, x.split(",")))) 
            if isinstance(x, str) and x.strip() else np.array([])
        )
    return df

def get_customer_embedding(customer_id):
    customers_df = fetch_embeddings("customers")
    customer_row = customers_df[customers_df["customer_id"] == customer_id]
    if customer_row.empty:
        return None
    
    customer_data = {
        "embedding": customer_row["embeddings"].values[0],
        "avg_order_value": customer_row["avg_order_value"].values[0],
        "location": customer_row["location"].values[0],
        "gender": customer_row["gender"].values[0],
        "season": customer_row["season"].values[0]
    }
    
    print(f"Customer Data: {customer_data}")
    return customer_data

def get_product_embeddings():
    products_df = fetch_embeddings("products")
    products_df = products_df[products_df["embeddings"].apply(lambda x: len(x) > 0)]
    return products_df

def compute_similarity(customer_embedding, product_embeddings):
    if len(customer_embedding) == 0 or len(product_embeddings) == 0:
        return np.array([])
    return cosine_similarity([customer_embedding], product_embeddings)[0]

def apply_filters(products_df, customer_data):
    print(f"Customer's Location: {customer_data['location']}")
    print(f"Customer's Season: {customer_data['season']}")
    filtered_products = products_df[
        (products_df["price"] <= customer_data["avg_order_value"] * 1.5) 
    ]
    print(f"Total Products: {len(products_df)}")
    print(f"Products After Price Filter: {len(filtered_products)}")
    
    return filtered_products

def boost_scores(filtered_products):
    filtered_products["final_score"] = (
        filtered_products["similarity"] * 0.7 +
        filtered_products["probability_of_recommendation"] * 0.2 +
        filtered_products["customer_review_sentiment_score"] * 0.1
    )
    return filtered_products

def recommend_products(customer_id, top_n=5):
    customer_data = get_customer_embedding(customer_id)
    if customer_data is None:
        return "Customer not found."
    
    if len(customer_data["embedding"]) == 0:
        return "No valid embeddings for customer."
    
    products_df = get_product_embeddings()
    filtered_products = apply_filters(products_df, customer_data)
    
    if filtered_products.empty:
        return "No matching products found."
    
    product_embeddings = np.stack(filtered_products["embeddings"].values)
    similarity_scores = compute_similarity(customer_data["embedding"], product_embeddings)
    filtered_products["similarity"] = similarity_scores
    
    recommendations = boost_scores(filtered_products)
    recommendations = recommendations.nlargest(top_n, "final_score")
    
    return recommendations[[
        "product_id", "category", "price", 
        "probability_of_recommendation", "final_score"
    ]]

if __name__ == "__main__":
    customer_id = "C2387"  
    top_n = 5
    
    try:
        recommendations = recommend_products(customer_id, top_n)
        if isinstance(recommendations, pd.DataFrame):
            print("\nRecommendations:\n", recommendations)
        else:
            print(recommendations)
    except Exception as e:
        print(f"Error: {str(e)}")