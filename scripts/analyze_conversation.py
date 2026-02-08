#!/usr/bin/env python3
"""
分析Claude对话记录并生成统计报告
"""
import json
import os
from datetime import datetime
from collections import Counter

def analyze_conversation(jsonl_file):
    """分析对话记录"""

    stats = {
        'total_messages': 0,
        'user_messages': 0,
        'assistant_messages': 0,
        'tool_calls': 0,
        'errors': 0,
        'tools_used': Counter(),
        'first_message': None,
        'last_message': None,
    }

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                msg_type = data.get('type', '')
                timestamp = data.get('timestamp', '')

                stats['total_messages'] += 1

                # 记录时间
                if timestamp:
                    if not stats['first_message']:
                        stats['first_message'] = timestamp
                    stats['last_message'] = timestamp

                # 统计消息类型
                if msg_type == 'user':
                    stats['user_messages'] += 1
                elif msg_type == 'assistant':
                    stats['assistant_messages'] += 1
                    # 统计工具调用（assistant消息中可能包含）
                    if 'content' in data:
                        content = data['content']
                        if isinstance(content, str):
                            # 查找工具调用标记 (简化的检测方式)
                            if '<tool_use>' in content or 'Tool' in content:
                                # 粗略估计工具调用
                                pass

                # 统计工具调用 (如果直接存在)
                if 'tool_use' in data or 'tool_calls' in data:
                    if 'tool_calls' in data:
                        for tool_call in data['tool_calls']:
                            tool_name = tool_call.get('name', 'unknown')
                            stats['tool_calls'] += 1
                            stats['tools_used'][tool_name] += 1

            except json.JSONDecodeError:
                pass
            except Exception:
                pass

    return stats

def format_duration(start, end):
    """格式化时间跨度"""
    if not start or not end:
        return "未知"

    try:
        dt_start = datetime.fromisoformat(start.replace('Z', '+00:00'))
        dt_end = datetime.fromisoformat(end.replace('Z', '+00:00'))
        duration = dt_end - dt_start

        hours = duration.total_seconds() / 3600
        if hours > 24:
            days = int(hours // 24)
            hours_remainder = int(hours % 24)
            return f"{days}天{hours_remainder}小时"
        else:
            return f"{int(hours)}小时"
    except:
        return "未知"

def generate_markdown_report(stats, output_file):
    """生成Markdown报告"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# AutoFacial 项目开发对话统计\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 总览
        f.write("## 总览\n\n")
        f.write(f"- **总消息数**: {stats['total_messages']:,}\n")
        f.write(f"- **用户消息**: {stats['user_messages']:,}\n")
        f.write(f"- **Claude回复**: {stats['assistant_messages']:,}\n")
        f.write(f"- **工具调用**: {stats['tool_calls']:,}\n")
        f.write(f"- **错误次数**: {stats['errors']:,}\n")

        if stats['first_message'] and stats['last_message']:
            f.write(f"- **开始时间**: {stats['first_message'][:19].replace('T', ' ')}\n")
            f.write(f"- **结束时间**: {stats['last_message'][:19].replace('T', ' ')}\n")
            duration = format_duration(stats['first_message'], stats['last_message'])
            f.write(f"- **时间跨度**: {duration}\n")

        f.write("\n")

        # 工具使用统计
        f.write("## 工具使用统计\n\n")
        f.write("| 工具名称 | 调用次数 |\n")
        f.write("|---------|----------|\n")

        for tool, count in stats['tools_used'].most_common():
            f.write(f"| `{tool}` | {count:,} |\n")

        f.write("\n")

        # 主要功能
        f.write("## 主要开发功能\n\n")
        f.write("本次开发会话实现了以下功能：\n\n")
        f.write("1. **自动演员匹配功能**\n")
        f.write("   - 重新聚类后自动匹配剧集演员信息\n")
        f.write("   - 修复embedding加载bug (pickle → numpy)\n")
        f.write("   - 测试验证：成功匹配2个演员\n\n")

        f.write("2. **系统截图和文档**\n")
        f.write("   - 添加4个系统界面截图\n")
        f.write("   - 优化README格式和内容\n\n")

        f.write("3. **代码推送到GitHub**\n")
        f.write("   - 清理大文件历史\n")
        f.write("   - 成功推送到 https://github.com/Earo24/auto_facial\n\n")

if __name__ == "__main__":
    jsonl_file = "HITL/conversation_history.jsonl"

    if not os.path.exists(jsonl_file):
        print(f"错误: 找不到对话记录文件 {jsonl_file}")
        exit(1)

    print("分析对话记录...")
    stats = analyze_conversation(jsonl_file)

    print(f"\n统计结果:")
    print(f"  总消息数: {stats['total_messages']:,}")
    print(f"  用户消息: {stats['user_messages']:,}")
    print(f"  Claude回复: {stats['assistant_messages']:,}")
    print(f"  工具调用: {stats['tool_calls']:,}")

    output_file = "HITL/CONVERSATION_STATS.md"
    generate_markdown_report(stats, output_file)

    print(f"\n✓ 统计报告已生成: {output_file}")
