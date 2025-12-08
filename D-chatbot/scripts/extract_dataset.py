import json
import re
from pathlib import Path

RAW_PATH = "data/raw/xrjb.txt"
OUT_PATH = "data/cleaned/strategy1.json"

TARGET_QQ = "480667648"   # 你要模仿的A的QQ号
CONTEXT_WINDOW = 2         # 取前面2句话当上下文

def parse_chat(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    messages = []
    i = 0

    # 匹配例如： 2019-07-22 15:55:37 Livermorium(1522261953)
    header_regex = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2})\s+(.+?)\((\d+)\)",
    re.UNICODE
)


    while i < len(lines):
        header_match = header_regex.match(lines[i])
        if header_match:
            timestamp, name, qq = header_match.groups()

            # 下一行才是内容
            if i + 1 < len(lines):
                content = lines[i + 1].strip()
            else:
                content = ""

            # 过滤掉图片、表情、空内容
            if content and content not in ["[图片]", "[表情]"]:
                messages.append({
                    "timestamp": timestamp,
                    "name": name,
                    "qq": qq,
                    "content": content,
                })

            i += 2  # 跳过正文
        else:
            i += 1

    return messages


def build_dataset(messages):
    dataset = []

    for idx, msg in enumerate(messages):
        if msg["qq"] != TARGET_QQ:
            continue  # 不是A的发言就跳过

        # 取上下文
        context_msgs = messages[max(0, idx - CONTEXT_WINDOW):idx]
        # 只保留内容，并用句号分隔
        context_text = "。".join([m["content"] for m in context_msgs])

        if not context_text.strip():
            context_text = "（无上下文）"

        dataset.append({
            "instruction": f"你现在模仿A（QQ号：{TARGET_QQ}）的语气回答。",
            "input": context_text,
            "output": msg["content"],
        })

    return dataset



def main():
    print("🔍 正在解析聊天记录...")
    msgs = parse_chat(RAW_PATH)

    print(f"总共解析到 {len(msgs)} 条有效消息")

    dataset = build_dataset(msgs)

    print(f"👉 最终生成 {len(dataset)} 条属于 {TARGET_QQ} 的训练数据")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dataset, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"✔ 数据已保存到 {OUT_PATH}")


if __name__ == "__main__":
    main()
