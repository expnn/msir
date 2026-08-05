import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path


def main():
    stub_file = Path("msir.pyi")

    # 1. 调用 maturin 生成原始存根，输出目标直接设为根目录 '.'
    print("正在通过 maturin 生成原始存根文件...")
    subprocess.run(("maturin", "generate-stubs", "--out", "."), check=True)  # ruff: ignore[start-process-with-partial-path]

    # 2. 读取刚刚生成的 pyi 内容
    if stub_file.exists():
        content = stub_file.read_text(encoding="utf-8")

        # 3. 在最顶部拼接缺少的 numpy 导入
        custom_imports = "import numpy as np\n\n"

        if "import numpy" not in content:
            fixed_content = custom_imports + content
            stub_file.write_text(fixed_content, encoding="utf-8")
            print(f"✅ 成功补全 NumPy 导入并保存到: {stub_file}")
        else:
            print("💡 存根文件中已存在 numpy 导入，跳过追加。")
    else:
        print(f"❌ 未在根目录下找到生成的 {stub_file}，请检查 Cargo.toml 中的库名称。")


if __name__ == "__main__":
    main()
