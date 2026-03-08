"""
RAG Engine V2 - 双路并行检索策略
同时利用 FAISS 向量库和 Neo4j 知识图谱来解答用户问答
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import jieba
import jieba.posseg as pseg
from neo4j import GraphDatabase
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 关系类型同义词映射（用户提问 → 数据库中的关系类型）
RELATION_SYNONYMS = {
    # ===== 声优/制作相关 =====
    "配音": ["配音", "声优", "CV"],
    "声优": ["配音", "声优", "CV"],
    "CV": ["配音", "声优", "CV"],
    "配过": ["配音", "声优", "CV"],
    "演过": ["配音", "声优", "CV"],
    "配音演员": ["配音", "声优", "CV"],
    "作者": ["作者", "原作", "漫画家", "作画"],
    "原作": ["原作", "作者", "漫画家"],
    "漫画家": ["作者", "原作", "漫画家"],
    "制作": ["制作", "动画制作", "制作公司", "动画公司"],
    "动画制作": ["制作", "动画制作", "制作公司"],
    "工作室": ["制作", "动画制作", "制作公司"],
    "导演": ["导演", "监督"],
    "监督": ["导演", "监督"],
    "编剧": ["编剧", "脚本", "系列构成"],
    "脚本": ["编剧", "脚本"],

    # ===== 恋爱关系 =====
    "男朋友": ["男朋友", "男友", "恋人", "喜欢的人", "对象", "男朋友"],
    "男友": ["男朋友", "男友", "恋人"],
    "女朋友": ["女朋友", "女友", "恋人", "喜欢的人", "对象"],
    "女友": ["女朋友", "女友", "恋人"],
    "恋人": ["恋人", "男朋友", "女朋友", "喜欢的人", "对象"],
    "喜欢的人": ["喜欢的人", "恋人", "男朋友", "女朋友", "暗恋对象"],
    "暗恋": ["暗恋", "喜欢的人", "暗恋对象"],
    "暗恋对象": ["暗恋", "喜欢的人", "暗恋对象"],
    "对象": ["对象", "恋人", "男朋友", "女朋友"],
    "情侣": ["情侣", "恋人", "夫妻"],
    "夫妻": ["夫妻", "配偶", "丈夫", "妻子"],
    "丈夫": ["丈夫", "老公", "配偶"],
    "妻子": ["妻子", "老婆", "配偶"],
    "老公": ["丈夫", "老公"],
    "老婆": ["妻子", "老婆"],
    "青梅竹马": ["青梅竹马", "幼驯染"],
    "幼驯染": ["青梅竹马", "幼驯染"],

    # ===== 家庭关系 =====
    "父亲": ["父亲", "爸爸", "老爸", "爹"],
    "爸爸": ["父亲", "爸爸", "老爸"],
    "母亲": ["母亲", "妈妈", "老妈", "娘"],
    "妈妈": ["母亲", "妈妈", "老妈"],
    "父母": ["父亲", "母亲", "父母"],
    "兄弟": ["兄弟", "哥哥", "弟弟", "兄", "弟"],
    "姐妹": ["姐妹", "姐姐", "妹妹", "姐", "妹"],
    "哥哥": ["哥哥", "兄", "哥哥"],
    "弟弟": ["弟弟", "弟"],
    "姐姐": ["姐姐", "姐"],
    "妹妹": ["妹妹", "妹"],
    "儿子": ["儿子", "孩子", "儿子"],
    "女儿": ["女儿", "孩子"],
    "孩子": ["儿子", "女儿", "孩子"],
    "子女": ["儿子", "女儿", "子女"],
    "爷爷": ["爷爷", "祖父"],
    "奶奶": ["奶奶", "祖母"],
    "外公": ["外公", "外祖父"],
    "外婆": ["外婆", "外祖母"],
    "叔叔": ["叔叔", "叔父"],
    "阿姨": ["阿姨", "婶婶"],
    "舅舅": ["舅舅", "舅父"],
    "舅妈": ["舅妈", "舅母"],
    "表亲": ["表亲", "表兄弟", "表姐妹"],
    "堂亲": ["堂亲", "堂兄弟", "堂姐妹"],
    "亲戚": ["亲戚", "亲属", "家人"],
    "家人": ["家人", "亲属", "家人"],
    "亲属": ["亲属", "家人", "亲戚"],

    # ===== 友情关系 =====
    "朋友": ["朋友", "好友", "挚友", "友人"],
    "好友": ["好友", "朋友", "挚友"],
    "挚友": ["挚友", "好友", "死党"],
    "死党": ["死党", "挚友", "好友"],
    "发小": ["发小", "青梅竹马", "儿时玩伴"],
    "玩伴": ["玩伴", "儿时玩伴", "青梅竹马"],
    "同学": ["同学", "同班同学", "同校"],
    "同桌": ["同桌", "同座"],
    "室友": ["室友", "同居人"],

    # ===== 战斗/冒险关系 =====
    "搭档": ["搭档", "同伴", "队友", "组合"],
    "同伴": ["同伴", "搭档", "队友", "伙伴"],
    "队友": ["队友", "搭档", "同伴"],
    "伙伴": ["伙伴", "同伴", "搭档"],
    "师父": ["师父", "师傅", "师傅", "师尊", "老师"],
    "师傅": ["师父", "师傅", "老师"],
    "师尊": ["师父", "师尊", "师傅"],
    "徒弟": ["徒弟", "弟子", "学生", "徒弟"],
    "弟子": ["徒弟", "弟子", "学生"],
    "学生": ["学生", "徒弟", "弟子"],
    "老师": ["老师", "教师", "师父", "导师"],
    "导师": ["导师", "老师", "指导者"],
    "后辈": ["后辈", "学弟", "学妹"],
    "前辈": ["前辈", "学长", "学姐"],
    "学长": ["学长", "前辈"],
    "学姐": ["学姐", "前辈"],
    "学弟": ["学弟", "后辈"],
    "学妹": ["学妹", "后辈"],

    # ===== 对立关系 =====
    "敌人": ["敌人", "对手", "宿敌", "仇人", "仇敌"],
    "对手": ["对手", "敌人", "宿敌", "竞争对手"],
    "宿敌": ["宿敌", "死敌", "宿敌", "宿命对手"],
    "死敌": ["死敌", "宿敌", "宿命对手"],
    "仇人": ["仇人", "仇敌", "敌人"],
    "竞争对手": ["竞争对手", "对手", "宿敌"],
    "反派": ["反派", "反派角色", "敌人", "BOSS"],
    "BOSS": ["BOSS", "boss", "最终BOSS", "反派"],

    # ===== 角色定位 =====
    "主要人物": ["主角", "主要人物", "主人公", "核心角色", "主要角色"],
    "主角": ["主角", "主要人物", "主人公", "核心角色"],
    "主人公": ["主人公", "主角", "主要人物"],
    "核心角色": ["核心角色", "主角", "主要人物"],
    "主要角色": ["主要角色", "主角", "主要人物"],
    "配角": ["配角", "主要配角", "次要角色"],
    "主要配角": ["主要配角", "配角"],
    "次要角色": ["次要角色", "配角"],
    "女主角": ["女主角", "女主", "女主人公"],
    "男主": ["男主", "男主角", "男主人公"],
    "男主角": ["男主角", "男主"],

    # ===== 其他常见关系 =====
    "仆人": ["仆人", "随从", "侍从"],
    "随从": ["随从", "仆人", "侍从"],
    "主人": ["主人", "雇主", "主公"],
    "雇主": ["雇主", "主人"],
    "领导": ["领导", "上司", "队长", "领袖"],
    "上司": ["上司", "领导", "上级"],
    "下属": ["下属", "部下", "手下"],
    "部下": ["部下", "下属", "手下"],
    "队长": ["队长", "首领", "领导"],
    "首领": ["首领", "领导", "头目"],
    "成员": ["成员", "队员", "组员"],
    "战友": ["战友", "战友", "战友"],
    "继承人": ["继承人", "继承者", "后继者"],
    "继承者": ["继承者", "继承人"],
    "化身": ["化身", "转世", " reincarnation"],
    "转世": ["转世", "化身", "转生"],
    "契约者": ["契约者", "契约对象", "契约主"],
    "召唤师": ["召唤师", "召唤者"],
    "召唤物": ["召唤物", "式神", "使魔"],
    "式神": ["式神", "召唤物", "使魔"],
    "使魔": ["使魔", "召唤物", "式神"],
    "宿主": ["宿主", "载体"],
    "寄宿者": ["寄宿者", "附身者"],
}


class RAGEngineV2:
    def __init__(self,
                 index_path="faiss_index_v2",
                 model_name="shibing624/text2vec-base-chinese",
                 api_url="http://localhost:1234/v1",
                 user_dict_path="user_dict.txt",
                 neo4j_uri="neo4j://127.0.0.1:7687",
                 neo4j_user="neo4j",
                 neo4j_password="12345678"):

        self.index_path = index_path
        self.model_name = model_name
        self.api_url = api_url
        self.user_dict_path = user_dict_path

        # Neo4j 配置
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # 初始化各组件
        self._init_jieba()
        self.embeddings = self._init_embeddings()
        self.vector_db = self._load_vector_db()
        self.neo4j_driver = self._init_neo4j()
        self.llm = self._init_llm()

    def _init_jieba(self):
        """初始化 jieba 分词，加载用户词典"""
        # 获取用户词典的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dict_path = os.path.join(script_dir, self.user_dict_path)

        if os.path.exists(dict_path):
            jieba.load_userdict(dict_path)
            print(f"[INFO] 已加载用户词典: {dict_path}")
        else:
            print(f"[WARNING] 用户词典不存在: {dict_path}")

    def _init_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'}
        )

    def _load_vector_db(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_index_path = os.path.join(script_dir, self.index_path)

        if not os.path.exists(full_index_path):
            raise FileNotFoundError(f"找不到索引文件: {full_index_path}")
        return FAISS.load_local(
            full_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def _init_neo4j(self):
        """初始化 Neo4j 连接"""
        try:
            driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            # 测试连接
            with driver.session() as session:
                session.run("RETURN 1")
            print(f"[INFO] Neo4j 连接成功: {self.neo4j_uri}")
            return driver
        except Exception as e:
            print(f"[WARNING] Neo4j 连接失败: {e}")
            return None

    def _init_llm(self):
        return ChatOpenAI(
            base_url=self.api_url,
            api_key="lm-studio",
            temperature=0.3
        )

    def extract_keywords_with_relations(self, query: str) -> tuple:
        """
        改进的关键词提取：识别实体名、动漫名、关系类型关键词
        返回: (entity_keywords, relation_keywords, query_type)
        """
        words = pseg.cut(query)

        # 实体关键词（名词类）
        entity_keywords = []
        # 关系类型关键词
        relation_keywords = []
        # 问题类型标识
        query_type = "relation"  # 默认是关系查询

        # ===== 智能判断问题类型 =====
        # 列表类问题：要求列举多个结果
        if any(x in query for x in ["有哪些", "有什么", "列出", "全部", "都有哪些", "都有什么", "都有谁"]):
            query_type = "list"
        # 声优类问题：配音相关
        elif any(x in query for x in ["配过", "配音", "声优", "CV", "演过", "配了", "配音演员"]):
            query_type = "voice_actor"
        # 关系类问题：询问特定关系
        elif any(x in query for x in ["是谁", "什么", "哪个", "是谁呀", "叫什么"]):
            query_type = "relation"
        # 隐式关系问题：如"水野茜的男朋友"（没有疑问词但有关系词）
        elif any(word in query for word in RELATION_SYNONYMS.keys()):
            query_type = "relation"

        # ===== 分词并分类关键词 =====
        for word, flag in words:
            # 过滤单字和虚词
            if len(word) <= 1:
                continue
            if flag in ['v', 'r', 'x', 'b', 'd', 'p', 'c', 'u', 'y', 'e', 'm', 'q']:
                continue

            # 检查是否是关系类型关键词
            if word in RELATION_SYNONYMS:
                relation_keywords.append(word)
            else:
                entity_keywords.append(word)

        # ===== 二次检查：从实体关键词中提取可能的关系词 =====
        # 有时候jieba分词会漏掉关系词，这里用字符串匹配补充
        remaining_entities = []
        for kw in entity_keywords:
            found_relation = False
            for rel_word in RELATION_SYNONYMS.keys():
                if rel_word in kw or kw in rel_word:
                    relation_keywords.append(rel_word)
                    found_relation = True
                    break
            if not found_relation:
                remaining_entities.append(kw)

        entity_keywords = remaining_entities

        print(f"[DEBUG] 问题类型: {query_type}")
        print(f"[DEBUG] 实体关键词: {entity_keywords}")
        print(f"[DEBUG] 关系关键词: {relation_keywords}")

        return entity_keywords, relation_keywords, query_type

    def fuzzy_match_relations(self, relation_keywords: list) -> tuple:
        """
        模糊匹配关系类型：当用户输入的关系词不在映射表中时，
        尝试找到最接近的关系类型
        返回: (expanded_relations, fuzzy_hints)
        """
        if not relation_keywords:
            return [], []

        expanded_relations = []
        fuzzy_hints = []  # 用于提示用户可能的关系

        for kw in relation_keywords:
            if kw in RELATION_SYNONYMS:
                expanded_relations.extend(RELATION_SYNONYMS[kw])
            else:
                # 尝试部分匹配
                matched = False
                for rel_key, rel_values in RELATION_SYNONYMS.items():
                    # 双向包含匹配
                    if kw in rel_key or rel_key in kw:
                        expanded_relations.extend(rel_values)
                        fuzzy_hints.append(rel_key)
                        matched = True
                        break
                    # 检查值列表
                    for rv in rel_values:
                        if kw in rv or rv in kw:
                            expanded_relations.extend(rel_values)
                            fuzzy_hints.append(rel_key)
                            matched = True
                            break
                    if matched:
                        break
                # 如果都没匹配上，保留原词用于数据库模糊查询
                if not matched:
                    expanded_relations.append(kw)

        return list(set(expanded_relations)), list(set(fuzzy_hints))

    def _build_voice_actor_query(self):
        """
        声优查询：支持双向检索
        场景：喜多村英梨配过什么角色？
        """
        return """
        // 1. 找动漫锚点（如果有）
        OPTIONAL MATCH (a_anchor:Anime)
        WHERE any(k IN $kws WHERE a_anchor.name CONTAINS k)
        WITH collect(distinct a_anchor) as anchors, $kws as kws, $rels as rels

        // 2. 双向检索：声优→角色 和 角色→声优
        CALL (anchors, kws, rels) {
            // 情况 A：关键词匹配声优名，找配音关系
            MATCH (e1:Entity)
            WHERE any(k IN kws WHERE e1.name CONTAINS k)
            MATCH (e1)-[r:RELATION]-(e2:Entity)
            WHERE size(rels) = 0 OR any(rel_type IN rels WHERE r.type CONTAINS rel_type OR rel_type CONTAINS r.type)
            MATCH (e1)-[:BELONG_TO]->(anime:Anime)
            MATCH (e2)-[:BELONG_TO]->(anime)
            RETURN e1, r, e2, anime.name as anime_tag

            UNION

            // 情况 B：关键词匹配角色名，找声优关系
            MATCH (e2:Entity)
            WHERE any(k IN kws WHERE e2.name CONTAINS k)
            MATCH (e1:Entity)-[r:RELATION]-(e2)
            WHERE size(rels) = 0 OR any(rel_type IN rels WHERE r.type CONTAINS rel_type OR rel_type CONTAINS r.type)
            MATCH (e1)-[:BELONG_TO]->(anime:Anime)
            MATCH (e2)-[:BELONG_TO]->(anime)
            RETURN e1, r, e2, anime.name as anime_tag
        }

        // 3. 智能评分：双向匹配 + 关系类型权重
        WITH e1, r, e2, anime_tag, kws, rels,
             CASE
                // 精确匹配优先级最高
                WHEN e1.name IN kws OR e2.name IN kws THEN 40
                // 实体名包含关键词（模糊匹配）
                WHEN (any(k IN kws WHERE e1.name CONTAINS k) AND size(rels) > 0) THEN 35
                WHEN (any(k IN kws WHERE e2.name CONTAINS k) AND size(rels) > 0) THEN 35
                WHEN any(k IN kws WHERE e1.name CONTAINS k) THEN 30
                WHEN any(k IN kws WHERE e2.name CONTAINS k) THEN 28
                // 关系类型匹配（声优查询中很重要）
                WHEN size(rels) > 0 AND any(rel_type IN rels WHERE r.type CONTAINS rel_type OR rel_type CONTAINS r.type) THEN 25
                // 动漫名匹配
                WHEN anime_tag IN kws THEN 20
                ELSE 1
             END AS score

        WHERE score >= 20

        RETURN DISTINCT e1.name as source, r.type as rel, e2.name as target, anime_tag, score
        ORDER BY score DESC, source, target
        LIMIT 20
        """

    def _build_list_query(self):
        """
        列表查询：返回动漫下的所有角色
        场景：进击的巨人有哪些主要人物？
        """
        return """
        // 1. 匹配动漫锚点
        MATCH (a:Anime)
        WHERE any(k IN $kws WHERE a.name CONTAINS k OR k CONTAINS a.name)
        WITH a, $kws as kws

        // 2. 检索该动漫下所有实体关系
        MATCH (e1:Entity)-[:BELONG_TO]->(a)
        MATCH (e1)-[r:RELATION]-(e2:Entity)
        MATCH (e2)-[:BELONG_TO]->(a)
        WITH e1, r, e2, a.name as anime_tag, kws

        // 3. 评分：优先显示重要角色关系
        WITH e1, r, e2, anime_tag, kws,
             CASE
                // 动漫名精确匹配
                WHEN anime_tag IN kws THEN 50
                // 主角/主要人物关键词
                WHEN any(k IN kws WHERE k IN ["主角", "主要人物", "主人公", "核心角色"]) THEN 40
                // 动漫名模糊匹配
                WHEN any(k IN kws WHERE anime_tag CONTAINS k) THEN 35
                // 关系类型匹配
                WHEN any(k IN kws WHERE r.type CONTAINS k) THEN 25
                ELSE 10
             END AS score

        WHERE score >= 10

        RETURN DISTINCT e1.name as source, r.type as rel, e2.name as target, anime_tag, score
        ORDER BY score DESC, source, target
        LIMIT 30
        """

    def _build_relation_query(self):
        """
        关系查询：特定角色的特定关系
        场景：水野茜的男朋友是谁？
        """
        return """
        // 1. 找动漫锚点（如果有）
        OPTIONAL MATCH (a_anchor:Anime)
        WHERE any(k IN $kws WHERE a_anchor.name CONTAINS k)
        WITH collect(distinct a_anchor) as anchors, $kws as kws, $rels as rels

        // 2. 双向检索关系
        CALL (anchors, kws, rels) {
            // 情况 A：通过动漫锚点检索
            UNWIND CASE WHEN size(anchors) > 0 THEN anchors ELSE [null] END AS a
            MATCH (e1:Entity)-[:BELONG_TO]->(anime:Anime)
            WHERE a IS null OR anime = a
            MATCH (e1)-[r:RELATION]-(e2:Entity)
            MATCH (e2)-[:BELONG_TO]->(anime)
            WHERE (size(kws) = 0 OR any(k IN kws WHERE e1.name CONTAINS k OR e2.name CONTAINS k))
              AND (size(rels) = 0 OR any(rel_type IN rels WHERE r.type CONTAINS rel_type OR rel_type CONTAINS r.type))
            RETURN e1, r, e2, anime.name as anime_tag

            UNION

            // 情况 B：直接通过关键词检索
            MATCH (e1:Entity)
            WHERE any(k IN kws WHERE e1.name CONTAINS k)
            MATCH (e1)-[r:RELATION]-(e2:Entity)
            MATCH (e1)-[:BELONG_TO]->(anime:Anime)
            MATCH (e2)-[:BELONG_TO]->(anime)
            WHERE size(rels) = 0 OR any(rel_type IN rels WHERE r.type CONTAINS rel_type OR rel_type CONTAINS r.type)
            RETURN e1, r, e2, anime.name as anime_tag
        }

        // 3. 智能评分：双向匹配 + 关系类型优先
        WITH e1, r, e2, anime_tag, kws, rels,
             CASE
                // 双向精确匹配（实体A或B精确匹配）
                WHEN e1.name IN kws THEN 45
                WHEN e2.name IN kws THEN 45
                // 实体模糊匹配 + 关系类型匹配
                WHEN (any(k IN kws WHERE e1.name CONTAINS k) AND size(rels) > 0) THEN 40
                WHEN (any(k IN kws WHERE e2.name CONTAINS k) AND size(rels) > 0) THEN 38
                // 关系类型精确匹配（重要！）
                WHEN size(rels) > 0 AND r.type IN rels THEN 35
                // 关系类型模糊匹配
                WHEN size(rels) > 0 AND any(rel_type IN rels WHERE r.type CONTAINS rel_type) THEN 32
                WHEN size(rels) > 0 AND any(rel_type IN rels WHERE rel_type CONTAINS r.type) THEN 30
                // 实体模糊匹配
                WHEN any(k IN kws WHERE e1.name CONTAINS k) THEN 28
                WHEN any(k IN kws WHERE e2.name CONTAINS k) THEN 25
                // 动漫名匹配
                WHEN anime_tag IN kws THEN 20
                ELSE 5
             END AS score

        WHERE score >= 20

        RETURN DISTINCT e1.name as source, r.type as rel, e2.name as target, anime_tag, score
        ORDER BY score DESC, source, target
        LIMIT 20
        """

    def _fallback_entity_search(self, session, entity_keywords: list) -> list:
        """
        降级查询：当精确查询无结果时，放宽条件只匹配实体名
        返回匹配到的实体的所有关系
        """
        fallback_cypher = """
        MATCH (e1:Entity)
        WHERE any(k IN $kws WHERE e1.name CONTAINS k)
        MATCH (e1)-[r:RELATION]-(e2:Entity)
        MATCH (e1)-[:BELONG_TO]->(anime:Anime)
        MATCH (e2)-[:BELONG_TO]->(anime)
        RETURN DISTINCT e1.name as source, r.type as rel, e2.name as target, anime.name as anime_tag
        """
        result = session.run(fallback_cypher, kws=entity_keywords)
        return list(result)

    def search_neo4j(self, entity_keywords: list, relation_keywords: list, query_type: str) -> str:
        """在 Neo4j 知识图谱中搜索相关实体和关系（改进版）"""
        if not self.neo4j_driver or not entity_keywords:
            return ""

        results = []
        seen_entities = set()

        # 获取扩展的关系类型（带模糊匹配）
        expanded_relations, fuzzy_hints = self.fuzzy_match_relations(relation_keywords)
        if expanded_relations:
            print(f"[DEBUG] 扩展关系类型: {expanded_relations}")
        if fuzzy_hints:
            print(f"[DEBUG] 模糊匹配提示: {fuzzy_hints}")

        with self.neo4j_driver.session() as session:
            # 根据问题类型选择不同的查询策略
            if query_type == "list":
                cypher = self._build_list_query()
            elif query_type == "voice_actor":
                cypher = self._build_voice_actor_query()
            else:
                cypher = self._build_relation_query()

            try:
                result = session.run(cypher,
                                    kws=entity_keywords,
                                    rels=expanded_relations)
                records = list(result)

                if records:
                    print(f"[DEBUG] Neo4j 命中结果 ({len(records)} 条)")
                    for rec in records:
                        u = rec['source']
                        r = rec['rel']
                        v = rec['target']
                        tag = rec['anime_tag'] if rec['anime_tag'] else "未知作品"

                        key = f"{u}-{r}-{v}"
                        if key not in seen_entities:
                            seen_entities.add(key)
                            results.append(f"[{tag}] {u} --{r}--> {v}")
                else:
                    print("[DEBUG] Neo4j 未命中，尝试降级查询...")
                    # 尝试降级查询：只匹配实体名，不限制关系类型
                    if entity_keywords:
                        fallback_records = self._fallback_entity_search(session, entity_keywords)
                        if fallback_records:
                            print(f"[DEBUG] 降级查询命中 ({len(fallback_records)} 条)")
                            for rec in fallback_records:
                                u = rec['source']
                                r = rec['rel']
                                v = rec['target']
                                tag = rec['anime_tag'] if rec['anime_tag'] else "未知作品"

                                key = f"{u}-{r}-{v}"
                                if key not in seen_entities:
                                    seen_entities.add(key)
                                    results.append(f"[{tag}] {u} --{r}--> {v}")

            except Exception as e:
                print(f"[DEBUG] Neo4j 查询错误: {e}")

        neo4j_context = "\n".join(results) if results else ""
        print(f"[DEBUG] Neo4j 检索到 {len(results)} 条关系")
        return neo4j_context

    def filter_core_keywords(self, keywords: list) -> list:
        """过滤核心关键词，用于向量检索"""
        # 过滤常见的疑问词和停用词
        stop_words = {'什么', '怎么', '如何', '为什么', '哪里', '谁', '哪个',
                      '吗', '呢', '吧', '的', '了', '是', '有', '在', '和',
                      '与', '或', '但', '如果', '虽然', '因为', '所以', '可以',
                      '能', '会', '应该', '需要', '请', '帮我', '告诉我', '问'}

        core_keywords = [kw for kw in keywords if kw not in stop_words and len(kw) > 1]
        print(f"[DEBUG] 核心关键词: {core_keywords}")
        return core_keywords

    def search_faiss(self, keywords: list, k: int = 3) -> str:
        """在 FAISS 向量库中搜索相关文档"""
        # 用关键词组合成查询语句
        query_text = " ".join(keywords)

        retriever = self.vector_db.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query_text)

        print(f"[DEBUG] FAISS 检索到 {len(docs)} 条片段")
        for i, doc in enumerate(docs):
            print(f"  - 片段{i}: {doc.page_content[:50]}...")

        return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, query: str) -> str:
        """双路并行检索并生成回答（改进版）"""
        # 1. 使用改进的关键词提取（区分实体和关系关键词）
        entity_keywords, relation_keywords, query_type = self.extract_keywords_with_relations(query)

        # 2. 过滤核心关键词（用于向量检索）
        core_keywords = self.filter_core_keywords(entity_keywords) if entity_keywords else []

        # 3. 并行检索（这里顺序执行，实际可改为并发）
        # Neo4j 检索（使用全部分词结果）
        neo4j_context = self.search_neo4j(entity_keywords, relation_keywords, query_type)

        # FAISS 检索（使用过滤后的核心关键词）
        faiss_context = self.search_faiss(core_keywords) if core_keywords else self.search_faiss(entity_keywords)

        # 4. 合并上下文
        combined_context = self._merge_contexts(neo4j_context, faiss_context)

        # 5. 生成回答
        response = self._generate_response(query, combined_context)

        return response

    def _merge_contexts(self, neo4j_context: str, faiss_context: str) -> str:
        """合并两个知识库的检索结果"""
        parts = []

        if neo4j_context.strip():
            parts.append("【知识图谱信息】\n" + neo4j_context)

        if faiss_context.strip():
            parts.append("【文档资料】\n" + faiss_context)

        return "\n\n".join(parts) if parts else "未找到相关资料"

    def _generate_response(self, query: str, context: str) -> str:
        """使用 LLM 生成回答"""
        prompt = ChatPromptTemplate.from_template("""
1.你是一个叫A同学的，精通二次元知识的，自嘲废宅风的AI助手，正在与用户进行交流。
  现在请结合以下提供的知识图谱相对高分信息和文档资料来回答问题。
2.禁止编造，如果查不到相关资料则回复不知道。

{context}

【用户问题】
{question}

A同学的回答：""")

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

    def close(self):
        """关闭资源连接"""
        if self.neo4j_driver:
            self.neo4j_driver.close()

    def __del__(self):
        self.close()


# 测试入口
if __name__ == "__main__":
    engine = RAGEngineV2()

    test_questions = [
        "鲁迪乌斯是谁？",
        "辉夜大小姐想让我告白的作者是谁？",
        "进击的巨人有什么主要角色？"
    ]

    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"问题: {q}")
        print(f"{'='*50}")
        answer = engine.invoke(q)
        print(f"\n回答: {answer}")

    engine.close()