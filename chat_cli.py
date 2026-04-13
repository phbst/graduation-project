import requests
import sqlite3
import re
import pandas as pd
import time
import json
import os

# ========== 配置 ==========
API_URL = "http://localhost:8000/v1/chat/completions"   # 你直接提供的接口
DB_PATH = "./funds_v3.db"
PROMPT_TEMPLATE_FILE = "infer.template"
DB_CONFIG_FILE = "config.json"

TEMPERATURE = 0
REQUEST_TIMEOUT = 30

# ========== 全局变量 ==========
current_table = None
db_config = None

# ========== 工具函数 ==========

def load_db_config():
    """加载数据库配置"""
    global db_config
    try:
        with open(DB_CONFIG_FILE, 'r', encoding='utf-8') as f:
            db_config = json.load(f)
        print(f"[INFO] 成功加载数据库配置，包含 {len(db_config)} 个表")
        return True
    except Exception as e:
        print(f"[ERROR] 加载数据库配置失败: {e}")
        return False

def show_table_menu():
    """显示表选择菜单"""
    if not db_config:
        print("[ERROR] 数据库配置未加载")
        return None
    
    print("\n=== 可用数据表 ===")
    tables = list(db_config.keys())
    for i, table in enumerate(tables, 1):
        print(f"{i}. {table}")
    
    while True:
        try:
            choice = input(f"\n请选择表 (1-{len(tables)}) 或输入 'q' 返回: ").strip()
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(tables):
                selected_table = tables[choice_num - 1]
                print(f"✅ 已选择表: {selected_table}")
                return selected_table
            else:
                print(f"[ERROR] 请输入 1-{len(tables)} 之间的数字")
        except ValueError:
            print("[ERROR] 请输入有效的数字")

def call_model_api(query: str, table_name: str = None) -> str:
    """调用大模型接口解析 SQL"""
    # 获取建表语句
    build_statement = ""
    if table_name and db_config and table_name in db_config:
        build_statement = db_config[table_name]["build"]
    elif current_table and db_config and current_table in db_config:
        build_statement = db_config[current_table]["build"]
    
    with open(PROMPT_TEMPLATE_FILE, encoding='utf-8') as f:
        prompt = f.read().format(query=query, build=build_statement)
        print(prompt)
    payload = {
        "model": "",  # 如果需要可填写模型名，否则留空
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE
    }
    try:
        start = time.time()
        resp = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        
        latency = time.time() - start
        print(f"[INFO] 模型响应耗时: {latency:.2f}s")

        return data['choices'][0]['message']['content']
    except Exception as e:
        print(f"[ERROR] 调用模型失败: {e}")
        return ""

def extract_sql(resp: str) -> str:
    """从模型返回文本中提取SQL"""
    if not resp:
        return ""
    text = str(resp)

    # 优先找```sql```代码块
    m = re.findall(r'```sql\s*(.*?)```', text, flags=re.S | re.I)
    if m:
        return m[0].strip()

    # 次选普通代码块
    m = re.findall(r'```\s*(.*?)```', text, flags=re.S)
    if m and any(k in m[0].upper() for k in ['SELECT','UPDATE','DELETE','INSERT']):
        return m[0].strip()

    # 否则直接返回整段
    return text.strip()

def fix_table_name(sql: str, table_name: str = None) -> str:
    """修正表名"""
    target_table = table_name or current_table
    if not target_table:
        # 原有的修正逻辑保持不变
        corrections = {
            'funds': 'Fund_products',
            'FundTable': 'Fund_products',
            'Fund_Table': 'Fund_products',
            'fund_products': 'Fund_products'
        }
        for wrong, right in corrections.items():
            sql = re.sub(r'\b' + re.escape(wrong) + r'\b', right, sql, flags=re.I)
        return sql
    
    # 根据当前选择的表进行修正
    common_table_variants = [
        'table', 'data', 'info', '表', target_table.lower(), target_table.upper()
    ]
    
    for variant in common_table_variants:
        sql = re.sub(r'\b' + re.escape(variant) + r'\b', target_table, sql, flags=re.I)
    
    return sql

def exec_sql(sql: str):
    """执行SQL并打印前5行"""
    if not sql:
        print("[WARN] 没有解析出SQL")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(sql)

        if sql.strip().upper().startswith('SELECT'):
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            df = pd.DataFrame(rows, columns=cols)

            if df.empty:
                print("✅ 查询成功，但无结果。")
            else:
                print("✅ 查询成功，前5行结果：")
                print(df.head(5).to_string(index=False))
                if len(df) > 5:
                    print(f"... (共 {len(df)} 行)")
        else:
            conn.commit()
            print(f"✅ SQL执行成功，受影响行数: {cur.rowcount}")

        conn.close()
    except Exception as e:
        print(f"[ERROR] SQL执行失败: {e}")

# ========== 主交互循环 ==========

def show_commands():
    """显示可用命令"""
    print("\n=== 可用命令 ===")
    print("exit/quit/q - 退出程序")
    print("table/tables - 选择数据表")
    print("current - 显示当前选择的表")
    print("help - 显示此帮助信息")
    print("其他输入 - 自然语言查询问题")

def main():
    global current_table
    
    print("=== SQL CLI 工具 ===")
    print("支持多表选择问答，查询JSON配置，动态填写prompt")
    
    # 加载数据库配置
    if not load_db_config():
        print("无法加载数据库配置，程序退出")
        return
    
    show_commands()
    
    while True:
        prompt_prefix = f"[{current_table}]" if current_table else "[未选择表]"
        query = input(f"\n{prompt_prefix} 请输入命令或查询问题> ").strip()
        
        if query.lower() in ["exit", "quit", "q"]:
            print("Bye!")
            break
        elif query.lower() in ["table", "tables"]:
            selected = show_table_menu()
            if selected:
                current_table = selected
            continue
        elif query.lower() == "current":
            if current_table:
                print(f"当前选择的表: {current_table}")
            else:
                print("未选择任何表")
            continue
        elif query.lower() == "help":
            show_commands()
            continue
        elif not query:
            continue

        # 检查是否选择了表
        if not current_table:
            print("[WARN] 请先选择一个数据表，输入 'table' 进行选择")
            continue

        # 调模型
        resp = call_model_api(query)
        if not resp:
            continue

        print("\n[模型原始输出]:")
        print(resp)

        # 提取SQL
        sql = extract_sql(resp)
        sql = fix_table_name(sql)

        print("\n[解析出的SQL]:")
        print(sql)

        # 执行SQL
        exec_sql(sql)


if __name__ == "__main__":
    main()

# import requests
# import sqlite3
# import re
# import pandas as pd
# import time

# # ========== 配置 ==========
# API_URL = "http://localhost:8001/v1/chat/completions"   # 你直接提供的接口
# DB_PATH = "../eval/funds_v3.db"
# PROMPT_TEMPLATE_FILE = "infer.template"

# TEMPERATURE = 0.1
# REQUEST_TIMEOUT = 30

# # ========== 工具函数 ==========

# def call_model_api(query: str) -> str:
#     """调用大模型接口解析 SQL"""
#     with open(PROMPT_TEMPLATE_FILE, encoding='utf-8') as f:
#         prompt = f.read().format(query=query)
#         print(prompt)
#     payload = {
#         "model": "",  # 如果需要可填写模型名，否则留空
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": TEMPERATURE
#     }
#     try:
#         start = time.time()
#         resp = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
#         resp.raise_for_status()
#         data = resp.json()
        
#         latency = time.time() - start
#         print(f"[INFO] 模型响应耗时: {latency:.2f}s")

#         return data['choices'][0]['message']['content']
#     except Exception as e:
#         print(f"[ERROR] 调用模型失败: {e}")
#         return ""

# def extract_sql(resp: str) -> str:
#     """从模型返回文本中提取SQL"""
#     if not resp:
#         return ""
#     text = str(resp)

#     # 优先找```sql```代码块
#     m = re.findall(r'```sql\s*(.*?)```', text, flags=re.S | re.I)
#     if m:
#         return m[0].strip()

#     # 次选普通代码块
#     m = re.findall(r'```\s*(.*?)```', text, flags=re.S)
#     if m and any(k in m[0].upper() for k in ['SELECT','UPDATE','DELETE','INSERT']):
#         return m[0].strip()

#     # 否则直接返回整段
#     return text.strip()

# def fix_table_name(sql: str) -> str:
#     """修正表名"""
#     corrections = {
#         'funds': 'Fund_products',
#         'FundTable': 'Fund_products',
#         'Fund_Table': 'Fund_products',
#         'fund_products': 'Fund_products'
#     }
#     for wrong, right in corrections.items():
#         sql = re.sub(r'\b' + re.escape(wrong) + r'\b', right, sql, flags=re.I)
#     return sql

# def exec_sql(sql: str):
#     """执行SQL并打印前5行"""
#     if not sql:
#         print("[WARN] 没有解析出SQL")
#         return

#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cur = conn.cursor()
#         cur.execute(sql)

#         if sql.strip().upper().startswith('SELECT'):
#             rows = cur.fetchall()
#             cols = [c[0] for c in cur.description]
#             df = pd.DataFrame(rows, columns=cols)

#             if df.empty:
#                 print("✅ 查询成功，但无结果。")
#             else:
#                 print("✅ 查询成功，前5行结果：")
#                 print(df.head(5).to_string(index=False))
#                 if len(df) > 5:
#                     print(f"... (共 {len(df)} 行)")
#         else:
#             conn.commit()
#             print(f"✅ SQL执行成功，受影响行数: {cur.rowcount}")

#         conn.close()
#     except Exception as e:
#         print(f"[ERROR] SQL执行失败: {e}")

# # ========== 主交互循环 ==========

# def main():
#     print("=== SQL CLI 工具 ===")
#     print("输入自然语言问题，大模型会生成SQL并执行")
#     print("输入 'exit' 退出")

#     while True:
#         query = input("\n请输入查询问题> ").strip()
#         if query.lower() in ["exit", "quit", "q"]:
#             print("Bye!")
#             break

#         # 调模型
#         resp = call_model_api(query)
#         if not resp:
#             continue

#         print("\n[模型原始输出]:")
#         print(resp)

#         # 提取SQL
#         sql = extract_sql(resp)
#         sql = fix_table_name(sql)

#         print("\n[解析出的SQL]:")
#         print(sql)

#         # 执行SQL
#         exec_sql(sql)


# if __name__ == "__main__":
#     main()
