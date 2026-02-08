#!/usr/bin/env python3
"""
将Claude对话记录从JSONL格式导出为Markdown格式
"""
import json
import os
from datetime import datetime

def export_conversation_to_markdown(jsonl_file, output_file):
    """导出对话记录为Markdown格式"""

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("# AutoFacial 项目开发对话记录\n\n")
        out.write(f"**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        out.write(f"**对话条数**: {len(lines)}\n\n")
        out.write("---\n\n")

        for i, line in enumerate(lines, 1):
            try:
                data = json.loads(line)

                # 获取消息类型
                msg_type = data.get('type', 'unknown')

                if msg_type == 'interaction':
                    # 用户消息
                    message = data.get('message', {})
                    content = message.get('content', '')
                    timestamp = data.get('timestamp', '')

                    if content:
                        out.write(f"## 用户消息 #{i}\n\n")
                        if timestamp:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            out.write(f"**时间**: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        out.write(f"{content}\n\n")
                        out.write("---\n\n")

                elif msg_type == 'working':
                    # 模型回复
                    content = data.get('content', '')
                    timestamp = data.get('timestamp', '')

                    if content:
                        out.write(f"## Claude 回复 #{i}\n\n")
                        if timestamp:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            out.write(f"**时间**: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        out.write(f"{content}\n\n")
                        out.write("---\n\n")

                elif msg_type == 'thinking':
                    # 思考过程（可选导出）
                    pass

                elif msg_type == 'error':
                    # 错误信息
                    error = data.get('error', '')
                    out.write(f"## 错误信息 #{i}\n\n")
                    out.write(f"```\n{error}\n```\n\n")
                    out.write("---\n\n")

            except json.JSONDecodeError as e:
                out.write(f"## 解析错误 (行 {i})\n\n")
                out.write(f"无法解析JSON: {e}\n\n")
                out.write("---\n\n")
            except Exception as e:
                out.write(f"## 处理错误 (行 {i})\n\n")
                out.write(f"处理错误: {e}\n\n")
                out.write("---\n\n")

    print(f"✓ 导出完成!")
    print(f"  输入文件: {jsonl_file}")
    print(f"  输出文件: {output_file}")
    print(f"  总条数: {len(lines)}")

    # 获取文件大小
    input_size = os.path.getsize(jsonl_file) / (1024 * 1024)
    output_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  输入大小: {input_size:.2f} MB")
    print(f"  输出大小: {output_size:.2f} MB")

if __name__ == "__main__":
    jsonl_file = "./conversation_history.jsonl"
    output_file = "./CONVERSATION_HISTORY.md"

    if not os.path.exists(jsonl_file):
        print(f"错误: 找不到对话记录文件 {jsonl_file}")
        exit(1)

    export_conversation_to_markdown(jsonl_file, output_file)
