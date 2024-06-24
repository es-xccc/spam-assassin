import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import flet as ft


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

# contents 包含所有檔案的內容，labels 包含對應的標籤

# 計算TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(contents)

# tfidf_matrix 是一個稀疏矩陣，包含了所有檔案的 TF-IDF 值
# labels 包含每個檔案的類別標籤

# 使用分層抽樣分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, stratify=labels, random_state=42)

# 創建隨機森林分類器實體
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 使用訓練集訓練模型
rf_classifier.fit(X_train, y_train)

# 使用測試集進行預測
y_pred = rf_classifier.predict(X_test)

# 計算準確率、精確率、召回率
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

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


def main(page: ft.Page):
    page.title = "垃圾郵件檢測器"
    page.padding = 10
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.width = 800  # 設置視窗寬度
    page.window.height = 600  # 設置視窗高度
    page.window.resizable = False  # 禁止調整視窗大小

    def pick_files_result(e: ft.FilePickerResultEvent):
        if e.files:
            selected_file.value = e.files[0].name
            selected_file.update()
            
            file_path = e.files[0].path
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            email_content.value = content
            email_content.update()
            check_button.disabled = False
            result_text.value = ""
        else:
            selected_file.value = "未選擇文件"
            email_content.value = ""
            check_button.disabled = True
            result_text.value = ""
        # 更新所有相關元件
        selected_file.update()
        email_content.update()
        check_button.update()
        result_text.update()

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_file = ft.Text(size=14, color=ft.colors.BLUE)
    email_content = ft.TextField(
        multiline=True,
        read_only=True,
        min_lines=15,
        max_lines=15,
        expand=True,
        border_color=ft.colors.OUTLINE,
        text_size=12
    )

    def check_spam(e):
        if not pick_files_dialog.result or not pick_files_dialog.result.files:
            return

        file_path = pick_files_dialog.result.files[0].path
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

        tfidf_vector = vectorizer.transform([content])
        prediction = rf_classifier.predict(tfidf_vector)[0]

        if prediction == 1:
            result_text.value = "這是垃圾郵件"
            result_text.color = ft.colors.RED
        else:
            result_text.value = "這不是垃圾郵件"
            result_text.color = ft.colors.GREEN
        result_text.update()

    check_button = ft.ElevatedButton("檢查", on_click=check_spam, disabled=True)
    result_text = ft.Text(size=14, weight=ft.FontWeight.BOLD)

    left_column = ft.Column([
        ft.Text("垃圾郵件檢測", size=20, weight=ft.FontWeight.BOLD),
        ft.Container(height=10),
        ft.Row([
            ft.ElevatedButton(
                "選擇郵件文件",
                icon=ft.icons.UPLOAD_FILE,
                on_click=lambda _: pick_files_dialog.pick_files(allow_multiple=False)
            ),
        ]),
        ft.Container(height=5),
        selected_file,
        ft.Container(height=10),
        check_button,
        ft.Container(height=10),
        result_text,
    ], alignment=ft.MainAxisAlignment.START, expand=1)

    right_column = ft.Column([
        ft.Text("郵件內容:", size=16, weight=ft.FontWeight.BOLD),
        ft.Container(height=5),
        email_content
    ], expand=2)

    page.overlay.append(pick_files_dialog)
    page.add(
        ft.Row([
            ft.Container(content=left_column, padding=10, border=ft.border.all(1, ft.colors.OUTLINE), border_radius=10, width=250),
            ft.Container(width=10),
            ft.Container(content=right_column, padding=10, border=ft.border.all(1, ft.colors.OUTLINE), border_radius=10, expand=True)
        ], expand=True)
    )

ft.app(target=main)