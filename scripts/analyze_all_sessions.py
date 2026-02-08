#!/usr/bin/env python3
"""
分析所有AutoFacial项目对话记录并生成汇总报告
"""
import json
import os
from datetime import datetime
from collections import Counter

def analyze_session(filepath):
    """分析单个会话"""
    stats = {
        'total_messages': 0,
        'user_messages': 0,
        'assistant_messages': 0,
        'first_message': None,
        'last_message': None,
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                msg_type = data.get('type', '')
                timestamp = data.get('timestamp', '')

                stats['total_messages'] += 1

                if timestamp:
                    if not stats['first_message']:
                        stats['first_message'] = timestamp
                    stats['last_message'] = timestamp

                if msg_type == 'user':
                    stats['user_messages'] += 1
                elif msg_type == 'assistant':
                    stats['assistant_messages'] += 1

            except:
                pass

    return stats

def format_time(timestamp):
    """格式化时间"""
    if not timestamp:
        return "N/A"
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%m-%d %H:%M')
    except:
        return "N/A"

def generate_summary_report():
    """生成汇总报告"""

    hitl_dir = "HITL"
    sessions = [
        ('session_0706_01.json', '2/7 早晨开发'),
        ('session_0707_02.json', '2/7 下午开发'),
        ('session_0707_03.json', '2/7 晚间短会'),
        ('session_0707_main.json', '2/7-2/8 主要会话'),
        ('session_0808_04.json', '2/8 最新会话'),
    ]

    all_stats = []

    for filename, desc in sessions:
        filepath = os.path.join(hitl_dir, filename)
        if os.path.exists(filepath):
            stats = analyze_session(filepath)
            stats['name'] = desc
            stats['file'] = filename
            size = os.path.getsize(filepath) / (1024 * 1024)
            stats['size_mb'] = size
            all_stats.append(stats)

    # 生成Markdown报告
    output_file = os.path.join(hitl_dir, "ALL_SESSIONS_STATS.md")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# AutoFacial 项目对话记录汇总\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 总览
        total_messages = sum(s['total_messages'] for s in all_stats)
        total_user = sum(s['user_messages'] for s in all_stats)
        total_assistant = sum(s['assistant_messages'] for s in all_stats)
        total_size = sum(s['size_mb'] for s in all_stats)

        f.write("## 总览\n\n")
        f.write(f"- **总会话数**: {len(all_stats)}\n")
        f.write(f"- **总消息数**: {total_messages:,}\n")
        f.write(f"- **用户消息**: {total_user:,}\n")
        f.write(f"- **Claude回复**: {total_assistant:,}\n")
        f.write(f"- **总数据量**: {total_size:.1f} MB\n\n")

        # 时间范围
        first = all_stats[0]['first_message']
        last = all_stats[-1]['last_message']
        if first and last:
            f.write(f"- **时间范围**: {format_time(first)} ~ {format_time(last)}\n\n")

        # 各会话详情
        f.write("## 各会话详情\n\n")
        f.write("| 会话 | 时间范围 | 消息数 | 用户 | 大小 |\n")
        f.write("|------|----------|--------|------|------|\n")

        for stats in all_stats:
            start = format_time(stats['first_message'])
            end = format_time(stats['last_message'])
            f.write(f"| {stats['name']} | {start} ~ {end} | {stats['total_messages']:,} | {stats['user_messages']} | {stats['size_mb']:.1f}M |\n")

        f.write("\n")

        # 文件列表
        f.write("## 文件列表\n\n")
        f.write("```\n")
        for stats in all_stats:
            f.write(f"{stats['file']}: {stats['name']}\n")
        f.write("```\n\n")

    print(f"✓ 汇总报告已生成: {output_file}")

    # 打印摘要
    print(f"\n统计摘要:")
    print(f"  总会话: {len(all_stats)}")
    print(f"  总消息: {total_messages:,}")
    print(f"  总大小: {total_size:.1f} MB")

if __name__ == "__main__":
    generate_summary_report()
