import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv(r'C:\Users\Asha\Documents\AIML\Log classification project\classification-logs\training\dataset\synthetic_logs.csv')
print(df)
print(df.source.unique())
print(df.target_label.unique())
#Clustering
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
embeddings = model.encode(df['log_message'].tolist())
print(embeddings)       
clustering = DBSCAN(eps=0.2, min_samples=1, metric='cosine').fit(embeddings)
df['cluster'] = clustering.labels_
print(df)
# Group by cluster to inspect patterns
clusters = df.groupby('cluster')['log_message'].apply(list)
sorted_clusters = clusters.sort_values(key=lambda x: x.map(len), ascending=False)
print("Clustered Patterns:")
for cluster_id, messages in sorted_clusters.items():
    if len(messages) > 10:
        print(f"Cluster {cluster_id}:")
        for msg in messages[:5]:
            print(f"  {msg}")
#Classification Stage 1: Regex
import re
def classify_with_regex(log_message):
    regex_patterns = {
        r"User User\d+ logged (in|out).": "User Action",
        r"Backup (started|ended) at .*": "System Notification",
        r"Backup completed successfully.": "System Notification",
        r"System updated to version .*": "System Notification",
        r"File .* uploaded successfully by user .*": "System Notification",
        r"Disk cleanup completed successfully.": "System Notification",
        r"System reboot initiated by user .*": "System Notification",
        r"Account with ID .* created by .*": "User Action"
    }
    for pattern, label in regex_patterns.items():
        if re.search(pattern, log_message):
            return label
    return None
print(classify_with_regex("User User123 logged in."))
print(classify_with_regex("System reboot initiated by user User179."))
print(classify_with_regex("Hey you, chill bro"))
# Apply regex classification
df['regex_label'] = df['log_message'].apply(lambda x: classify_with_regex(x))
df[df['regex_label'].notnull()]
print(df)
#Classification Stage 2: Classification Using Embeddings
df_non_regex = df[df['regex_label'].isnull()].copy()
print(df_non_regex.shape)
print(df_non_regex['target_label'].value_counts()[df_non_regex['target_label'].value_counts()<=5].index.tolist())
df_legacy = df_non_regex[df_non_regex.source=="LegacyCRM"]
print(df_legacy)
df_non_legacy = df_non_regex[df_non_regex.source!="LegacyCRM"]
print(df_non_legacy)
print(df_non_legacy.shape)
#Using BERT Embedding
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
embeddings_filtered = model.encode(df_non_legacy['log_message'].tolist())
len(embeddings_filtered)
#Model Training
X = embeddings_filtered
y = df_non_legacy['target_label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
#Save Model
joblib.dump(clf, '../models/log_classifier.joblib')
