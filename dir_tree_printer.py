import os


def print_tree_summary(startpath, max_level=3):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        if level >= max_level:
            continue
        indent = '│   ' * level + '├── '
        print(f"{indent}{os.path.basename(root)}/ ({len(files)} files)")
        subindent = '│   ' * (level + 1)
        for f in files[:5]:  # show only a few files
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... +{len(files) - 5} more files")


print_tree_summary('.', max_level=3)
