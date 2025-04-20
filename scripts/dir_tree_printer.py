# dir_tree_printer.py
import os


def human_readable_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def print_tree_summary(startpath, max_files_per_dir=5, depth=0):
    if not os.path.isdir(startpath):
        return

    entries = list(os.scandir(startpath))
    files = sorted([entry.name for entry in entries if entry.is_file()])
    dirs = sorted([entry for entry in entries if entry.is_dir()], key=lambda d: d.name)

    indent = '    ' * depth
    total_size = sum(os.path.getsize(os.path.join(startpath, f)) for f in files)
    print(f"{indent}{os.path.basename(startpath)}/ ({len(files)} files, {human_readable_size(total_size)})")

    for f in files[:max_files_per_dir]:
        display_name = f if len(f) <= 30 else f[:30] + "..."
        print(f"{indent}    {display_name}")
    if len(files) > max_files_per_dir:
        print(f"{indent}    ... +{len(files) - max_files_per_dir} more files")

    for d in dirs:
        print_tree_summary(d.path, max_files_per_dir, depth + 1)


print_tree_summary('../', max_files_per_dir=20)
