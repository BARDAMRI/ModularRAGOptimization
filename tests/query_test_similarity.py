import numpy as np
from llama_index.core.schema import MetadataMode

from config import INDEX_SOURCE_URL
from modules.indexer import load_vector_db
from utility.embedding_utils import get_query_vector


def embed_node_like_llamaindex(node, embedding_model):
    """
    מחקה בדיוק את התהליך של LlamaIndex ליצירת embeddings
    """
    # 1. קבל את הטקסט כמו ש-LlamaIndex עושה - עם metadata
    text_to_embed = node.get_content(metadata_mode=MetadataMode.EMBED)

    # 2. השתמש בפונקציה הנכונה לקידוד documents (לא queries)
    try:
        # נסה עם get_text_embedding (הפונקציה הסטנדרטית לדוקומנטים)
        embedding = embedding_model.get_text_embedding(text_to_embed)
    except Exception as e:
        print(f"⚠️  שגיאה בקידוד: {e}")
        # fallback - נסה בלי פרמטרים נוספים
        embedding = embedding_model.get_text_embedding(text_to_embed)

    # 3. המר לנמפי array
    embedding = np.array(embedding)

    # 4. בדוק אם המודל עושה normalization אוטומטי
    embedding_norm = np.linalg.norm(embedding)
    print(f"נורמה של הווקטור הגולמי: {embedding_norm:.6f}")

    # 5. normalize רק אם המודל לא עושה זאת בעצמו
    if embedding_norm > 1.1 or embedding_norm < 0.9:  # לא מנורמל
        embedding = normalize(embedding)
        print("✅ ביצעתי normalization ידני")
    else:
        print("✅ הווקטור כבר מנורמל")

    return embedding


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def analyze_text_differences(node):
    """
    בודק הבדלים בין הטקסט הגולמי לטקסט המעובד
    """
    original_text = node.text

    # טקסט עם metadata כמו ש-LlamaIndex משתמש
    embed_text = node.get_content(metadata_mode=MetadataMode.EMBED)

    print("🔍 ניתוח הטקסט")
    print("-" * 50)
    print(f"אורך טקסט מקורי: {len(original_text)}")
    print(f"אורך טקסט לקידוד: {len(embed_text)}")

    if original_text != embed_text:
        print("⚠️  הטקסט השתנה!")
        print("טקסט מקורי (100 תווים ראשונים):")
        print(repr(original_text[:100]))
        print("טקסט לקידוד (100 תווים ראשונים):")
        print(repr(embed_text[:100]))

        # בדוק metadata
        if hasattr(node, 'metadata') and node.metadata:
            print(f"Metadata קיים: {node.metadata}")
    else:
        print("✅ הטקסט זהה")

    return embed_text


# Load the vector database and embedding model
vector_db, embedding_model = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)

# בדוק את סוג המודל
print(f"🔍 מידע על המודל")
print("-" * 50)
print(f"סוג המודל: {type(embedding_model)}")
print(f"שם המודל: {getattr(embedding_model, 'model_name', 'לא ידוע')}")

# רשימת methods זמינים
methods = [method for method in dir(embedding_model) if 'embed' in method.lower()]
print(f"Methods זמינים: {methods}")

# Access internal vector store
vector_store = getattr(vector_db, "_vector_store", None)

# Prepare a query
query = "This is a sample query."
query_vector = normalize(get_query_vector(query, embedding_model))

# Query the VectorStoreIndex
nodes = vector_db.as_retriever(similarity_top_k=1).retrieve(query)
node_with_score = nodes[0]
retrieved_node = node_with_score.node
retrieved_score = node_with_score.score
node_id = retrieved_node.node_id

# === TEXT ANALYSIS ===
processed_text = analyze_text_differences(retrieved_node)

# === EMBEDDING ANALYSIS ===
print("\n🔍 ניתוח Embeddings")
print("-" * 50)

# Get stored vector from the index
stored_vector = normalize(np.array(vector_store.data.embedding_dict[node_id]))
print(f"ווקטור שמור - נורמה: {np.linalg.norm(np.array(vector_store.data.embedding_dict[node_id])):.6f}")

# Re-encode using LlamaIndex's exact method
reencoded_vector = embed_node_like_llamaindex(retrieved_node, embedding_model)

# Re-encode using original text (for comparison)
original_text = retrieved_node.text
reencoded_original = normalize(np.array(embedding_model.get_text_embedding(original_text)))

# Cosine similarities
cosine_vs_stored = np.dot(query_vector, stored_vector)
cosine_vs_reencoded = np.dot(query_vector, reencoded_vector)
cosine_vs_original = np.dot(query_vector, reencoded_original)

# Distances
distance_llamaindex = np.linalg.norm(reencoded_vector - stored_vector)
distance_original = np.linalg.norm(reencoded_original - stored_vector)

print(f"\n📊 תוצאות השוואה:")
print(f"Cosine - stored vector:           {retrieved_score:.6f}")
print(f"Cosine - manual stored:           {cosine_vs_stored:.6f}")
print(f"Cosine - LlamaIndex style:        {cosine_vs_reencoded:.6f}")
print(f"Cosine - original text only:      {cosine_vs_original:.6f}")

print(f"\n📏 מרחקים:")
print(f"||LlamaIndex style - stored||:    {distance_llamaindex:.6f}")
print(f"||Original text - stored||:       {distance_original:.6f}")

# בדיקת הצלחה
if distance_llamaindex < 1e-5:
    print("✅ הצלחנו! הווקטורים זהים")
elif distance_llamaindex < distance_original:
    print("🔶 השיפור חלקי - יש עדיין הבדל קטן")
else:
    print("❌ עדיין יש בעיה - צריך לחקור עוד")

# מידע נוסף לדיבוג
print(f"\nמימדי ווקטורים:")
print(f"Stored: {stored_vector.shape}")
print(f"Re-encoded: {reencoded_vector.shape}")
print(f"דוגמת ערכים (5 ראשונים):")
print(f"Stored:     {stored_vector[:5]}")
print(f"Re-encoded: {reencoded_vector[:5]}")
