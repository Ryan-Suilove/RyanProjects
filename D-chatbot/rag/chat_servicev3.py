from rag_enginev3 import RAGEngineV3

def start_interactive_chat():
    try:
        # 初始化双路检索引擎（V3优化版：支持智能重排序+metadata利用）
        engine = RAGEngineV3()
        print("\n--- A同学已上线！(双路检索模式V3：FAISS + Neo4j) ---")
        print("--- 优化特性：智能重排序 + Header信息利用 + 语义完整保留 ---")
        print("--- 输入 'exit' 或 'quit' 退出 ---\n")

        while True:
            user_input = input("你: ")
            if user_input.lower() in ['exit', 'quit']:
                break

            response = engine.invoke(user_input)
            print(f"\n助手: {response}\n")

    except Exception as e:
        print(f"启动失败: {e}")
    finally:
        # 确保关闭连接
        if 'engine' in locals():
            engine.close()

if __name__ == "__main__":
    start_interactive_chat()