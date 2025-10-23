import os
import shutil

# 获取vllm包的安装路径
try:
    import vllm
except ImportError:
    raise ImportError("vllm package not installed")

vllm_path = os.path.dirname(vllm.__file__)
print(f"vllm package path: {vllm_path}")

# 定义目标路径
target_dir = os.path.join(vllm_path, "model_executor", "models")
target_file = os.path.join(target_dir, "cosyvoice2.py")

# 复制模型文件
file_path = os.path.abspath(__file__)
print(f"Current file path: {file_path}")

model_file = os.path.join(os.path.dirname(file_path), "vllm_use_cosyvoice2_model.py")

if not os.path.exists(model_file):
    raise FileNotFoundError(f"Source file {model_file} not found")

shutil.copy(model_file, target_file)
print(f"Copied {model_file} to {target_file}")

# 修改registry.py文件
registry_path = os.path.join(target_dir, "registry.py")
new_entry = '    "CosyVoice2Model": ("cosyvoice2", "CosyVoice2Model"),  # noqa: E501\n'

# 读取并修改文件内容
with open(registry_path, "r") as f:
    lines = f.readlines()

# 检查是否已存在条目
entry_exists = any("CosyVoice2Model" in line for line in lines)

if not entry_exists:
    # 寻找插入位置
    insert_pos = None
    for i, line in enumerate(lines):
        # if line.strip().startswith("**_FALLBACK_MODEL"):
        if line.strip().startswith('**_EMBEDDING_MODELS'):
            insert_pos = i + 1
            break

    if insert_pos is None:
        raise ValueError("Could not find insertion point in registry.py")

    # 插入新条目
    lines.insert(insert_pos, new_entry)

    # 写回文件
    with open(registry_path, "w") as f:
        f.writelines(lines)
    print("Successfully updated registry.py")
else:
    print("Entry already exists in registry.py, skipping modification")

print("All operations completed successfully!")