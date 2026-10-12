"""
BeautifulSoup 解析 HTML 表格示例
以万用表文档中的"规格"表格为例，演示矩阵投影 + 语义转译
"""

from bs4 import BeautifulSoup

# ===================== 1. 原始 HTML 表格（带 rowspan） =====================

html = """
<table>
<tr><td>功能</td><td>量程</td><td>分辨率</td><td>精确度</td></tr>
<tr><td rowspan="5">直流电压</td><td>200mV</td><td>0.1mV</td><td rowspan="3">± (0.5% reading + 2 digits)</td></tr>
<tr><td>2000mV</td><td>1mV</td></tr>
<tr><td>20V</td><td>0.01V</td></tr>
<tr><td>200V</td><td>0.1V</td><td rowspan="2">± (0.8% reading + 2 digits)</td></tr>
<tr><td>600V</td><td>1V</td></tr>
<tr><td rowspan="2">交流电压</td><td>200V</td><td>0.1V</td><td rowspan="2">± (1.2% reading + 10 digits)</td></tr>
<tr><td>600V</td><td>1V</td></tr>
</table>
"""

# ===================== 2. 解析 HTML =====================

soup = BeautifulSoup(html, "html.parser")
table = soup.find("table") # 找表格
rows = table.find_all("tr") #  找所有行

# ===================== 3. 矩阵投影（核心） =====================
# 目标：把带 rowspan/colspan 的表格展开成一个规整的二维数组

# 3.1 先确定表格的行数和列数
num_rows = len(rows)
num_cols = max(
    sum(int(td.get("colspan", 1)) for td in row.find_all("td"))
    for row in rows
)

# 3.2 创建空的二维数组
grid = [[None] * num_cols for _ in range(num_rows)]

# 3.3 遍历每个单元格，填充到 grid 中
for row_idx, row in enumerate(rows):
    col_idx = 0
    for td in row.find_all("td"):
        # 找到当前行中第一个空位
        while col_idx < num_cols and grid[row_idx][col_idx] is not None:
            col_idx += 1

        # 获取 rowspan 和 colspan
        rowspan = int(td.get("rowspan", 1))
        colspan = int(td.get("colspan", 1))
        text = td.get_text(strip=True)

        # 向下、向右物理填充
        for r in range(rowspan):
            for c in range(colspan):
                if row_idx + r < num_rows and col_idx + c < num_cols:
                    grid[row_idx + r][col_idx + c] = text

        col_idx += colspan

# ===================== 4. 打印矩阵结果 =====================

print("=" * 70)
print("【矩阵投影结果】展开后的完整二维数组：")
print("=" * 70)

for row in grid:
    print(row)

# ===================== 5. 语义转译（降维） =====================
# 第一行是表头，后续每一行转成自然语言

print()
print("=" * 70)
print("【语义转译结果】拍平为自然语言：")
print("=" * 70)

headers = grid[0]

for row in grid[1:]:
    # 把每一行的表头和值拼起来
    parts = []
    for header, value in zip(headers, row):
        parts.append(f"{header}:{value}")

    sentence = "- 【" + "】【".join(parts) + "】"
    print(sentence)