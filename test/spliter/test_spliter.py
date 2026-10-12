from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "苹果的颜色是红色的[SEP]香蕉的颜色是黄色的[SEP]橘子的颜色是橙色的"

splitter_false = RecursiveCharacterTextSplitter(
    chunk_size=1,
    chunk_overlap=0,
    keep_separator=False
)
print(splitter_false.split_text(''))
