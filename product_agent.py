import pandas as pd
import sqlite3
from ollama import Client

def clean_list_str(s):
    s = s.replace(" ", "").replace("'", '"') 
    return s.strip()  
def preprocess_product_text(row):
    try:
        similar_products_str = clean_list_str(row["Similar_Product_List"])
        similar_products = eval(similar_products_str)  
    except Exception as e:
        print(f"Error parsing Similar_Product_List for {row['Product_ID']}: {e}")
        similar_products = []
    text = f"{row['Category']} {row['Subcategory']} {row['Brand']} " \
           f"{' '.join(similar_products)}"  
    return text.strip() 
def generate_embeddings(df):
    client = Client()
    embeddings = []
    for text in df["text_features"]:
        if not text.strip():
            embeddings.append([])
            continue
        try:
            response = client.embed("all-minilm:33m-l12-v2-fp16", text)
            embedding = response["embeddings"][0] if "embeddings" in response else []
        except Exception as e:
            print(f"Error generating embedding for text '{text}': {e}")
            embedding = []
        embeddings.append(embedding)
    df["embeddings"] = embeddings
    return df
def save_to_sqlite(df, db_name, table_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            category TEXT,
            subcategory TEXT,
            price REAL,
            brand TEXT,
            average_rating_of_similar_products REAL,
            product_rating REAL,
            customer_review_sentiment_score REAL,
            holiday TEXT,
            season TEXT,
            geographical_location TEXT,
            similar_product_list TEXT,  -- Stored as stringified list
            probability_of_recommendation REAL,
            text_features TEXT,
            embeddings TEXT        -- Comma-separated embeddings
        )
    ''')
    for _, row in df.iterrows():
        embeddings_list = row["embeddings"]
        embeddings_str = ",".join(map(str, embeddings_list)) if isinstance(embeddings_list, list) else ""
        cursor.execute('''
            INSERT OR REPLACE INTO products (
                product_id, category, subcategory, price, brand, 
                average_rating_of_similar_products, product_rating, 
                customer_review_sentiment_score, holiday, season, 
                geographical_location, similar_product_list, 
                probability_of_recommendation, text_features, embeddings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row["Product_ID"],
            row["Category"],
            row["Subcategory"],
            row["Price"],
            row["Brand"],
            row["Average_Rating_of_Similar_Products"],
            row["Product_Rating"],
            row["Customer_Review_Sentiment_Score"],
            row["Holiday"],
            row["Season"],
            row["Geographical_Location"],
            str(row["Similar_Product_List"]),
            row["Probability_of_Recommendation"],
            row["text_features"],
            embeddings_str
        ))
    conn.commit()
    conn.close()
    print(f"Product data saved to {db_name}")
if __name__ == "__main__":
    input_csv = "syn-dataset/product_recommendation_data.csv"
    df_products = pd.read_csv(input_csv)
    df_products["text_features"] = df_products.apply(preprocess_product_text, axis=1)
    df_products = generate_embeddings(df_products)
    save_to_sqlite(df_products, "ecommerce.db", "products")