import os
import json
from glob import glob
from openai import OpenAI
from neo4j import GraphDatabase

API_KEY = "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"
MODEL_ID = "qwen/qwen3.5-plus-02-15"
BASE_URL = "https://openrouter.ai/api/v1"

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(SCRIPT_DIR, "..", "wiki", "anime_knowledge_base")

class KnowledgeGraphBuilder:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def extract_triplets(self, wiki_content: str, anime_name: str) -> list:
        prompt = f"""你是一个知识图谱构建专家。请从以下动漫《{anime_name}》的资料中提取实体-关系-实体三元组。

要求：
1. 提取所有重要的人物、组织、作品等实体
2. 关系需要明确且有意义，例如：
   - 角色A - 女主角 - 作品名
   - 角色A - 声优 - 声优名
   - 角色A - 恋人 - 角色B
   - 角色A - 朋友 - 角色B
   - 作品 - 制作公司 - 公司名
   - 作品 - 原作 - 作者名
   - 作品 - 类型 - 类型名
   - 角色 - 身份 - 身份描述
3. 尽可能提取完整的关系，包括人物关系、作品信息等
4. 每个三元组一行，格式为：实体1|关系|实体2

动漫资料：
{wiki_content}

请直接输出三元组，每行一个，格式为：实体1|关系|实体2
不要输出其他内容。"""
        try:
            response = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()
            triplets = []
            for line in result.split('\n'):
                line = line.strip()
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        entity1 = parts[0].strip()
                        relation = parts[1].strip()
                        entity2 = '|'.join(parts[2:]).strip()
                        if entity1 and relation and entity2:
                            triplets.append((entity1, relation, entity2))
            return triplets
        except Exception as e:
            print(f"提取三元组失败: {e}")
            return []

    def create_entity_and_relation(self, tx, entity1: str, relation: str, entity2: str, anime_name: str):
        """
        核心修改：建立星型拓扑结构
        """
        # 1. 创建动漫根节点
        # 2. 创建两个实体节点
        # 3. 创建实体间关系
        # 4. 创建实体到动漫的归属边 (BELONG_TO)
        query = """
        MERGE (a:Anime {name: $anime_name})
        MERGE (e1:Entity {name: $entity1})
        MERGE (e2:Entity {name: $entity2})
        
        // 建立实体间的业务逻辑关系
        MERGE (e1)-[r:RELATION {type: $relation}]->(e2)
        
        // 建立实体到动漫的溯源归属关系（核心关键！）
        MERGE (e1)-[:BELONG_TO]->(a)
        MERGE (e2)-[:BELONG_TO]->(a)
        """
        tx.run(query, entity1=entity1, entity2=entity2, relation=relation, anime_name=anime_name)

    def save_to_neo4j(self, triplets: list, anime_name: str):
        """
        保存时传入 anime_name
        """
        with self.driver.session() as session:
            for entity1, relation, entity2 in triplets:
                try:
                    session.execute_write(self.create_entity_and_relation, entity1, relation, entity2, anime_name)
                    print(f"  已存: [{anime_name}] {entity1} - {relation} - {entity2}")
                except Exception as e:
                    print(f"  保存失败: {entity1}, 错误: {e}")

    def process_all_wiki_files(self):
        wiki_files = glob(os.path.join(KNOWLEDGE_BASE_PATH, "*.md"))
        print(f"共发现 {len(wiki_files)} 个文件")

        for idx, wiki_file in enumerate(wiki_files, 1):
            anime_name = os.path.splitext(os.path.basename(wiki_file))[0]
            print(f"\n[{idx}/{len(wiki_files)}] 处理作品: {anime_name}")

            try:
                with open(wiki_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                triplets = self.extract_triplets(content, anime_name)
                if triplets:
                    # 传入当前作品名以建立归属关系
                    self.save_to_neo4j(triplets, anime_name)
                    print(f"  作品《{anime_name}》拓扑构建完成")

            except Exception as e:
                print(f"  处理失败: {e}")

def main():
    builder = KnowledgeGraphBuilder()
    try:
        builder.process_all_wiki_files()
    finally:
        builder.close()

if __name__ == "__main__":
    main()