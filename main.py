import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def read_files_from_directory(directory_path, label):
    """讀取指定目錄下所有檔案的內容，並為每個檔案分配標籤"""
    files_content = []
    labels = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            files_content.append(file.read())
            labels.append(label)
    return files_content, labels

# 定義檔案路徑及其對應標籤
paths_and_labels = [
    ('./easy_ham', 0),  # easy_ham 標記為 0
    ('./hard_ham', 0),  # hard_ham 也標記為 0
    ('./spam_2', 1)     # spam_2 標記為 1
]

# 讀取每個路徑下的檔案內容及標籤
contents = []
labels = []
for path, label in paths_and_labels:
    path_contents, path_labels = read_files_from_directory(path, label)
    contents.extend(path_contents)
    labels.extend(path_labels)

# 現在，`contents` 包含所有檔案的內容，`labels` 包含對應的標籤

# 計算TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(contents)

# `tfidf_matrix` 是一個稀疏矩陣，包含了所有檔案的TF-IDF值
# `labels` 包含每個檔案的類別標籤

# 使用分層抽樣分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, stratify=labels, random_state=42)

# 創建朴素貝葉斯分類器實體
nb_classifier = MultinomialNB()

# 使用訓練集訓練模型
nb_classifier.fit(X_train, y_train)

# 使用測試集進行預測
y_pred = nb_classifier.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 計算混淆矩陣
conf_matrix = confusion_matrix(y_test, y_pred)
labels_map = {0: 'Ham', 1: 'Spam'}
conf_matrix_df = pd.DataFrame(conf_matrix, index=[labels_map[i] for i in range(len(labels_map))], columns=[labels_map[i] for i in range(len(labels_map))])

# 使用Seaborn繪製混淆矩陣的熱圖
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')