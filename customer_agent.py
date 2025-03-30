import pandas as pd
import sqlite3
from ollama import Client

def preprocess_text_features(row):
    """
    Combine customer browsing and purchase history into a clean string.
    Example: "Books Biography Brand C Biography Comics"
    """
    try:
        browsing = eval(row["Browsing_History"])
        purchases = eval(row["Purchase_History"])
    except Exception as e:
        print(f"Error parsing lists for {row['Customer_ID']}: {e}")
        browsing = []
        purchases = []
    
    combined_text = " ".join(browsing + purchases)
    return combined_text.strip()

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
    
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            customer_id TEXT PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            location TEXT,
            browsing_history TEXT,
            purchase_history TEXT,
            customer_segment TEXT,
            avg_order_value REAL,
            holiday TEXT,
            season TEXT,
            text_features TEXT,
            embeddings TEXT
        )
    ''')
    
    for _, row in df.iterrows():
        embeddings_str = ",".join(map(str, row["embeddings"])) if isinstance(row["embeddings"], list) else ""
        cursor.execute(f'''
            INSERT OR REPLACE INTO {table_name} (
                customer_id, age, gender, location, browsing_history, 
                purchase_history, customer_segment, avg_order_value, 
                holiday, season, text_features, embeddings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row["Customer_ID"],
            row["Age"],
            row["Gender"],
            row["Location"],
            str(row["Browsing_History"]),
            str(row["Purchase_History"]),
            row["Customer_Segment"],
            row["Avg_Order_Value"],
            row["Holiday"],
            row["Season"],
            row["text_features"],
            embeddings_str
        ))
    conn.commit()
    conn.close()
    print(f"Customer data saved to {db_name}")

if __name__ == "__main__":
    input_csv = "syn-dataset/customer_data_collection.csv"
    df_customers = pd.read_csv(input_csv)
    
    # Preprocess text features (combine browsing and purchase history)
    df_customers["text_features"] = df_customers.apply(preprocess_text_features, axis=1)
    
    # Generate embeddings
    df_customers = generate_embeddings(df_customers)
    save_to_sqlite(df_customers, "ecommerce.db", "customers")