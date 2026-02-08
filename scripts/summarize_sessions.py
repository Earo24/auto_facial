#!/usr/bin/env python3
"""
分析AutoFacial项目各对话会话的主要内容
"""
import json
import os
from datetime import datetime
from collections import Counter

def extract_user_messages(filepath, limit=50):
    """提取用户消息用于分析"""
    messages = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('type') == 'user':
                    content = data.get('message', {}).get('content', '')
                    if content and len(content.strip()) > 0:
                        messages.append(content)
                        if len(messages) >= limit:
                            break
            except:
                pass
    return messages

def analyze_session_keywords(filepath):
    """分析会话关键词"""
    user_messages = extract_user_messages(filepath, limit=100)

    keywords = []
    for msg in user_messages:
        # 提取中文关键词
        words = msg.split()
        keywords.extend([w for w in words if len(w) >= 2])

    # 统计高频词
    word_count = Counter(keywords)

    # 过滤常见词
    common_words = {'的', '是', '我', '你', '了', '在', '和', '有', '不', '要', '这个', '怎么', '那', '就', '吗', '吧', '啊'}
    for word in common_words:
        word_count.pop(word, None)

    return word_count.most_common(10)

def generate_session_summary():
    """生成会话汇总"""

    sessions = [
        {
            'file': 'HITL/session_0706_01.json',
            'name': '2/7 早晨开发',
            'time_range': '06:54-15:29'
        },
        {
            'file': 'HITL/session_0707_02.json',
            'name': '2/7 下午开发',
            'time_range': '15:31-18:09'
        },
        {
            'file': 'HITL/session_0707_03.json',
            'name': '2/7 晚间短会',
            'time_range': '18:30-18:51'
        },
        {
            'file': 'HITL/session_0707_main.json',
            'name': '2/7-2/8 主要会话',
            'time_range': '18:11-02:25'
        },
        {
            'file': 'HITL/session_0808_04.json',
            'name': '2/8 最新会话',
            'time_range': '02:27-02:30'
        },
    ]

    # 输出Markdown
    output = []
    output.append("# AutoFacial 项目对话记录详细分析\n\n")
    output.append(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    output.append("---\n\n")

    output.append("## 各会话详细分析\n\n")

    for session in sessions:
        filepath = session['file']
        if not os.path.exists(filepath):
            continue

        output.append(f"### {session['name']}\n\n")
        output.append(f"**时间**: 2月7日 {session['time_range']}\n\n")

        # 获取用户消息
        user_msgs = extract_user_messages(filepath, limit=20)

        if len(user_msgs) > 0:
            output.append("**用户主要操作**:\n\n")
            for i, msg in enumerate(user_msgs[:10], 1):
                # 简化显示
                if len(msg) > 100:
                    msg = msg[:97] + "..."
                output.append(f"{i}. {msg}\n")
            output.append("\n")

        # 分析关键词
        keywords = analyze_session_keywords(filepath)
        if keywords:
            output.append("**关键词**:\n\n")
            for word, count in keywords[:5]:
                output.append(f"- {word} ({count}次)\n")
            output.append("\n")

        output.append("---\n\n")

    return ''.join(output)

if __name__ == "__main__":
    summary = generate_session_summary()

    output_file = "HITL/SESSIONS_DETAIL.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"✓ 详细分析已生成: {output_file}")
