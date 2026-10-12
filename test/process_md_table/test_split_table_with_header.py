def split_table_with_header(table_text: str, max_rows_per_chunk: int = 5):
    """切分表格，并为每个子块续传表头"""
    lines = table_text.strip().split("\n")

    # 前两行：表头 + 分隔线（如 |---|---|）
    header = lines[0]
    separator = lines[1]
    data_rows = lines[2:]

    chunks = []
    for i in range(0, len(data_rows), max_rows_per_chunk):
        batch = data_rows[i:i + max_rows_per_chunk]
        # 每个子块都拼上表头，保证 LLM 能看懂列含义
        chunk = "\n".join([header, separator] + batch)
        chunks.append(chunk)

    return chunks


if __name__ == '__main__':

    table = """| 功能   | 量程   | 精确度              |
    |----------|-------|---------------------|
    | 直流电压 | 200mV  | ± (0.5% + 2 digits) |
    | 直流电压 | 20V    | ± (0.5% + 2 digits) |
    | 交流电压 | 200V   | ± (1.2% + 10 digits)|
    | 交流电压 | 600V   | ± (1.2% + 10 digits)|"""

    for i, chunk in enumerate(split_table_with_header(table, max_rows_per_chunk=2)):
        print(f"--- Chunk {i + 1} ---")
        print(chunk)
        print()
