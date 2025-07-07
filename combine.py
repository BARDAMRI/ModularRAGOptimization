import os

root_dir = "/Users/bardamri/PycharmProjects/ModularRAGOptimization"
output_file_path = "combined_project.py"

# רשימת תיקיות שאנחנו רוצים להתעלם מהן
excluded_dirs = {"__pycache__", ".venv", "env", ".git", ".idea", "build", "dist", "tests", "user_query_datasets"}

python_files = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    # סינון תיקיות לא רלוונטיות (in-place)
    dirnames[:] = [d for d in dirnames if d not in excluded_dirs]

    for filename in filenames:
        if filename.endswith(".py") and filename != os.path.basename(output_file_path):
            full_path = os.path.join(dirpath, filename)
            python_files.append(full_path)

# כתיבת הקבצים המאושרים לקובץ אחד
with open(output_file_path, "w", encoding="utf-8") as outfile:
    for file_path in python_files:
        rel_path = os.path.relpath(file_path, root_dir)
        outfile.write(f"# === File: {rel_path} ===\n")
        try:
            with open(file_path, "r", encoding="utf-8") as infile:
                content = infile.read()
        except UnicodeDecodeError:
            print(f"⚠️  קובץ לא נקרא ב-UTF-8: {file_path}, עובר ל-latin-1")
            with open(file_path, "r", encoding="latin-1") as infile:
                content = infile.read()
        outfile.write(content)
        outfile.write("\n\n")

print(f"✅ נוצר קובץ מאוחד רק עם הקוד שלך: {output_file_path}")