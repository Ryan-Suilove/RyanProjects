import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DATA_PATH = r"C:\Users\lty20\PycharmProjects\RyanProjects\D-chatbot\wiki\anime_knowledge_base"
SAVE_PATH = "faiss_index_v3"
MODEL_NAME = "shibing624/text2vec-base-chinese"

# Markdown 标题分割配置：按一级、二级、三级标题分割
HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]

def build_vector_store():
    # 1. 初始化分割器
    # MarkdownHeaderTextSplitter：按标题层级分割，保留文档语义结构
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False  # 保留标题在内容中，便于检索时理解上下文
    )
    # RecursiveCharacterTextSplitter：对过大的 chunk 进行二次分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )

    print("正在初始化 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    all_processed_chunks = []

    # 2. 遍历文件夹处理文档
    print(f"正在逐个处理文档: {DATA_PATH} ...")
    if not os.path.exists(DATA_PATH):
        print(f"错误：路径不存在 {DATA_PATH}")
        return

    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".md")]

    for file_name in files:
        file_path = os.path.join(DATA_PATH, file_name)
        anime_title = file_name.replace(".md", "")

        try:
            # 加载单个文件
            loader = TextLoader(file_path, encoding='utf-8')
            single_document = loader.load()[0]

            # 先按 Markdown 标题分割
            md_chunks = md_splitter.split_text(single_document.page_content)

            # 对每个 md_chunk 检查大小，过大的进行二次分割
            for md_chunk in md_chunks:
                if len(md_chunk.page_content) > 800:
                    # 过大 chunk 进行二次分割
                    sub_chunks = text_splitter.split_text(md_chunk.page_content)
                    for sub_chunk in sub_chunks:
                        # 合并原始 metadata 和来源信息
                        enriched_content = f"【资料所属动漫：{anime_title}】\n{sub_chunk}"
                        new_doc = Document(
                            page_content=enriched_content,
                            metadata={**md_chunk.metadata, "source": anime_title}
                        )
                        all_processed_chunks.append(new_doc)
                else:
                    # 直接使用
                    enriched_content = f"【资料所属动漫：{anime_title}】\n{md_chunk.page_content}"
                    new_doc = Document(
                        page_content=enriched_content,
                        metadata={**md_chunk.metadata, "source": anime_title}
                    )
                    all_processed_chunks.append(new_doc)

            print(f"  - 已处理: {anime_title} (切分为 {len(md_chunks)} 段)")

        except Exception as e:
            print(f"  - 处理文件 {file_name} 时出错: {e}")

    if not all_processed_chunks:
        print("未发现有效文本段，取消索引构建。")
        return

    print(f"\n全部处理完成！总计获得 {len(all_processed_chunks)} 个带标签文本段")

    # 3. 创建向量库并保存
    print("正在生成向量并构建索引 (这可能需要几分钟)...")
    vector_db = FAISS.from_documents(all_processed_chunks, embeddings)
    vector_db.save_local(SAVE_PATH)
    print(f"恭喜！向量库已成功保存至: {SAVE_PATH}")

if __name__ == "__main__":
    build_vector_store()