import re


def isolate_tables(md_content: str):
    """将 Markdown 表格从正文中隔离出来"""
    # 匹配标准 Markdown 表格（以 | 开头的连续行）
    table_pattern = re.compile(r'(\n?(?:\|[^\n]+\|\n)+)', re.MULTILINE)

    tables = []
    text_parts = []
    last_end = 0

    for match in table_pattern.finditer(md_content):
        # 收集表格前的正文
        text_parts.append(md_content[last_end:match.start()])
        # 收集表格（作为独立 Chunk）
        tables.append(match.group())
        last_end = match.end()

    # 收集最后一段正文
    text_parts.append(md_content[last_end:])

    return text_parts, tables  # 正文送切分器，表格直接作为独立 Chunk


if __name__ == '__main__':

    md_content = """# 万用表规格说明

这是一份数字万用表的技术文档，以下是核心规格参数。

## 规格参数

| 功能     | 量程   | 分辨率 | 精确度              |
|----------|--------|--------|---------------------|
| 直流电压 | 200mV  | 0.1mV  | ± (0.5% + 2 digits) |
| 直流电压 | 20V    | 0.01V  | ± (0.5% + 2 digits) |
| 交流电压 | 200V   | 0.1V   | ± (1.2% + 10 digits)|
| 交流电压 | 600V   | 1V     | ± (1.2% + 10 digits)|

以上参数在18°C至28°C环境下测得。

## 电池测试标准

| 电池类型 | 良好    | 较弱        | 坏的   |
|----------|---------|-------------|--------|
| 9V电池   | >8.2V  | 7.2至8.2V   | <7.2V  |
| 1.5V电池 | >1.35V | 1.22至1.35V | <1.22V |

请按照以上标准判断电池状态。
"""

    text_parts, tables = isolate_tables(md_content)

    print("=" * 50)
    print(f"共提取到 {len(tables)} 个表格，{len(text_parts)} 段正文")
    print("=" * 50)

    for i, part in enumerate(text_parts):
        print(f"\n--- 正文 {i + 1} ---")
        print(part.strip() if part.strip() else "(空)")

    for i, table in enumerate(tables):
        print(f"\n--- 表格 {i + 1}（独立 Chunk） ---")
        print(table.strip())