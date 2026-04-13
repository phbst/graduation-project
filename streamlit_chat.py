import re

import altair as alt
import pandas as pd
import requests
import streamlit as st

# ========== 配置 ==========
API_BASE_URL = "http://127.0.0.1:8999"
CHAT_AVATAR_USER = "🧑‍💼"
CHAT_AVATAR_AI = "🤖"

# ========== 页面配置 ==========
st.set_page_config(
    page_title="NL2SQL Tool",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 样式CSS ==========
st.markdown("""
<style>
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] * {
        display: none !important;
        visibility: hidden !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        visibility: visible !important;
        width: 260px !important;
    }

    [data-testid="stSidebar"] {
        display: block !important;
        width: 260px !important;
        min-width: 260px !important;
    }

    .main .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }

    header[data-testid="stHeader"] {
        display: none;
    }

    .custom-title {
        font-size: 1.5rem !important;
        font-weight: 700;
        margin-bottom: 1rem !important;
        color: #1e3c72;
    }

    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .status-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 0.4rem 0.6rem;
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
        font-size: 0.85rem;
        color: #166534;
    }

    .error-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 6px;
        padding: 0.4rem 0.6rem;
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
        font-size: 0.85rem;
        color: #991b1b;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .sql-box {
        background: #f8f9fa;
        border-left: 5px solid #007bff;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


# ========== 辅助函数 ==========
def suggest_table_name(file_name: str) -> str:
    stem = file_name.rsplit(".", 1)[0].strip().lower() if file_name else "new_table"
    stem = re.sub(r"\W+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "new_table"
    if stem[0].isdigit():
        stem = f"table_{stem}"
    return stem


@st.cache_data(ttl=10)
def fetch_tables():
    try:
        response = requests.get(f"{API_BASE_URL}/tables")
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=10)
def fetch_models():
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


def query_api(query, table_names=None, table_name=None, model_name=None):
    try:
        payload = {"query": query}
        if table_names:
            payload["table_names"] = table_names
        elif table_name:
            payload["table_name"] = table_name
        if model_name:
            payload["model_name"] = model_name
        return requests.post(f"{API_BASE_URL}/query", json=payload).json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def fetch_chart_suggestion(query, sql, data, columns=None):
    try:
        payload = {
            "query": query,
            "sql": sql,
            "data": data,
            "columns": columns or [],
        }
        return requests.post(f"{API_BASE_URL}/chart_suggestion", json=payload).json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_raw_sql(sql):
    try:
        return requests.post(f"{API_BASE_URL}/execute_raw_sql", json={"sql": sql}).json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def fetch_table_preview(table_name):
    try:
        return requests.get(f"{API_BASE_URL}/table_preview/{table_name}").json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def fetch_table_schema(table_name):
    try:
        return requests.get(f"{API_BASE_URL}/tables/{table_name}/schema").json()
    except Exception:
        return None


def import_excel_file(uploaded_file, table_name, sheet_name=None, if_exists="replace"):
    try:
        uploaded_file.seek(0)
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        }
        data = {
            "table_name": table_name,
            "sheet_name": sheet_name or "",
            "if_exists": if_exists,
        }
        return requests.post(f"{API_BASE_URL}/import_excel", files=files, data=data).json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_table_api(table_name):
    try:
        return requests.delete(f"{API_BASE_URL}/tables/{table_name}").json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def render_chart(chart_config, data):
    if not chart_config or not data:
        return

    chart_type = chart_config.get("chart_type")
    x_field = chart_config.get("x_field")
    y_field = chart_config.get("y_field")
    series_field = chart_config.get("series_field")
    title = chart_config.get("title") or "结果可视化"

    if not x_field or not y_field:
        return

    df = pd.DataFrame(data)
    if x_field not in df.columns or y_field not in df.columns:
        return

    color = alt.Color(f"{series_field}:N") if series_field and series_field in df.columns else alt.value("#667eea")

    if chart_type == "bar":
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{x_field}:N", sort="-y"),
            y=alt.Y(f"{y_field}:Q"),
            color=color,
            tooltip=list(df.columns),
        ).properties(title=title, height=360)
        st.altair_chart(chart, use_container_width=True)
    elif chart_type == "line":
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X(f"{x_field}:N"),
            y=alt.Y(f"{y_field}:Q"),
            color=color,
            tooltip=list(df.columns),
        ).properties(title=title, height=360)
        st.altair_chart(chart, use_container_width=True)
    elif chart_type == "scatter":
        chart = alt.Chart(df).mark_circle(size=90).encode(
            x=alt.X(f"{x_field}:Q" if pd.api.types.is_numeric_dtype(df[x_field]) else f"{x_field}:N"),
            y=alt.Y(f"{y_field}:Q"),
            color=color,
            tooltip=list(df.columns),
        ).properties(title=title, height=360)
        st.altair_chart(chart, use_container_width=True)
    elif chart_type == "pie":
        pie_chart = alt.Chart(df).mark_arc().encode(
            theta=alt.Theta(f"{y_field}:Q"),
            color=alt.Color(f"{x_field}:N"),
            tooltip=list(df.columns),
        ).properties(title=title, height=360)
        st.altair_chart(pie_chart, use_container_width=True)


# ========== 初始化状态 ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
if "multi_table_mode" not in st.session_state:
    st.session_state.multi_table_mode = False


# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h2>⚙️ 控制台</h2></div>', unsafe_allow_html=True)
    page = st.radio("功能模块:", ["💬 智能对话", "⌨️ SQL 运行器", "📂 数据库浏览", "📥 导入表格", "🗑️ 删除表格"])

    health = check_api_health()
    if health:
        st.markdown(
            f'''<div class="status-box"><strong>✅ 系统在线</strong><br>表: {health.get('tables_loaded', 0)} | 模型: {health.get('models_loaded', 0)}</div>''',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="error-box"><strong>❌ API 离线</strong></div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

    if page == "💬 智能对话":
        st.markdown("### 📊 数据表")
        multi_mode = st.checkbox("多表模式", value=st.session_state.multi_table_mode)
        st.session_state.multi_table_mode = multi_mode

        tables_data = fetch_tables()
        if tables_data and tables_data.get("success"):
            table_options = tables_data["tables"]
            if multi_mode:
                st.session_state.selected_tables = st.multiselect("选择关联表:", table_options, key="multi_select")
            else:
                st.session_state.selected_table = st.selectbox("选择目标表:", table_options, key="single_select")

        st.markdown("### 🧠 AI 模型")
        models_data = fetch_models()
        if models_data and models_data.get("success"):
            st.session_state.selected_model = st.selectbox("选择模型:", list(models_data["models"].keys()))

        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ========== 主界面逻辑 ==========
if page == "💬 智能对话":
    st.markdown('<div class="custom-title">💬 自然语言 SQL 查询</div>', unsafe_allow_html=True)

    col_chat, col_status = st.columns([3, 1])

    with col_chat:
        for message in st.session_state.messages[-2:]:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>{CHAT_AVATAR_USER} 用户:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
            else:
                if "sql" in message:
                    st.markdown(f'<div class="sql-box"><strong>🔍 生成 SQL:</strong><br><code>{message["sql"]}</code></div>', unsafe_allow_html=True)
                if "data" in message and message["data"]:
                    st.dataframe(pd.DataFrame(message["data"]), use_container_width=True)
                if "chart" in message and message["chart"] and "data" in message and message["data"]:
                    render_chart(message["chart"], message["data"])
                if "error" in message:
                    st.error(message["error"])

        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area("请输入问题:", placeholder="例如：查询最近三天的订单", height=100)
            c1, c2 = st.columns([1, 1])
            submit_clicked = c1.form_submit_button("🚀 提交查询", use_container_width=True)
            example_clicked = c2.form_submit_button("💡 示例查询", use_container_width=True)

        final_query = None
        if example_clicked:
            final_query = "查询表内的前5条数据"
        elif submit_clicked and user_input:
            final_query = user_input

        if final_query:
            if st.session_state.multi_table_mode and not st.session_state.get("selected_tables"):
                st.error("请选择至少一个数据表")
            elif not st.session_state.multi_table_mode and not st.session_state.get("selected_table"):
                st.error("请选择一个数据表")
            else:
                st.session_state.messages = [{"role": "user", "content": final_query}]
                with st.spinner("AI 正在思考..."):
                    if st.session_state.multi_table_mode:
                        res = query_api(final_query, table_names=st.session_state.selected_tables, model_name=st.session_state.get("selected_model"))
                    else:
                        res = query_api(final_query, table_name=st.session_state.selected_table, model_name=st.session_state.get("selected_model"))

                    if res.get("success"):
                        chart = None
                        if res.get("data"):
                            chart_res = fetch_chart_suggestion(
                                final_query,
                                res.get("sql"),
                                res.get("data"),
                                res.get("columns"),
                            )
                            if chart_res.get("success") and chart_res.get("should_visualize"):
                                chart = chart_res.get("chart")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "sql": res.get("sql"),
                            "data": res.get("data"),
                            "chart": chart,
                        })
                    else:
                        st.session_state.messages.append({"role": "assistant", "error": res.get("error")})
                st.rerun()

    with col_status:
        st.subheader("🛠️ 配置预览")
        if st.session_state.multi_table_mode:
            st.write(f"**关联表:** `{', '.join(st.session_state.get('selected_tables', []))}`")
        else:
            st.write(f"**当前表:** `{st.session_state.get('selected_table')}`")
        st.write(f"**当前模型:** `{st.session_state.get('selected_model')}`")

elif page == "⌨️ SQL 运行器":
    st.markdown('<div class="custom-title">⌨️ SQL Playground</div>', unsafe_allow_html=True)
    sql_text = st.text_area("输入原生 SQL:", placeholder="SELECT * FROM table LIMIT 10;", height=200)
    if st.button("▶️ 立即运行", type="primary"):
        if sql_text:
            with st.spinner("执行中..."):
                res = execute_raw_sql(sql_text)
                if res.get("success"):
                    st.success(f"执行成功 (行数: {res.get('total_rows')})")
                    if res.get("data"):
                        st.dataframe(pd.DataFrame(res["data"]), use_container_width=True)
                else:
                    st.error(f"SQL 报错: {res.get('error')}")

elif page == "📂 数据库浏览":
    st.markdown('<div class="custom-title">📂 Database Explorer</div>', unsafe_allow_html=True)
    tables_res = fetch_tables()
    if tables_res and tables_res.get("success"):
        target = st.selectbox("选择表:", tables_res["tables"])
        if target:
            t1, t2 = st.tabs(["🏗️ 结构", "📊 预览"])
            with t1:
                schema = fetch_table_schema(target)
                if schema:
                    st.code(schema.get("build_statement"), language="sql")
            with t2:
                preview = fetch_table_preview(target)
                if preview.get("success"):
                    st.dataframe(pd.DataFrame(preview.get("data", [])), use_container_width=True)
                else:
                    st.error(preview.get("error") or "表预览失败")

elif page == "📥 导入表格":
    st.markdown('<div class="custom-title">📥 导入表格</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("拖入或选择本地文件", type=["xlsx", "xls", "csv"])
    default_name = suggest_table_name(uploaded_file.name) if uploaded_file else ""

    with st.form(key="import_form"):
        table_name = st.text_input("表名称", value=default_name, placeholder="例如：computer_products")
        sheet_name = st.text_input("工作表名称（Excel 可选）", placeholder="留空则读取第一个 sheet")
        if_exists = st.selectbox("写入模式", ["replace", "append", "fail"], index=0)
        submit_import = st.form_submit_button("📤 开始导入", use_container_width=True)

    if submit_import:
        if not uploaded_file:
            st.error("请先上传文件")
        elif not table_name.strip():
            st.error("请先填写表名称")
        else:
            with st.spinner("正在导入表格..."):
                result = import_excel_file(uploaded_file, table_name.strip(), sheet_name.strip(), if_exists)
            if result.get("success"):
                st.success(f"导入成功，新表名：{result.get('table_name')}，共 {result.get('total_rows')} 行")
                if result.get("columns"):
                    st.write("字段:", ", ".join(result["columns"]))
                if result.get("build_statement"):
                    st.code(result.get("build_statement"), language="sql")
                st.cache_data.clear()
                if not st.session_state.multi_table_mode:
                    st.session_state.selected_table = result.get("table_name")
                st.rerun()
            else:
                st.error(result.get("error") or "导入失败")

elif page == "🗑️ 删除表格":
    st.markdown('<div class="custom-title">🗑️ 删除表格</div>', unsafe_allow_html=True)
    tables_res = fetch_tables()
    if tables_res and tables_res.get("success") and tables_res.get("tables"):
        target = st.selectbox("选择要删除的表:", tables_res["tables"])
        confirm_name = st.text_input("请输入表名确认删除", placeholder="请再次输入上面的表名")
        if st.button("🗑️ 确认删除", type="primary", use_container_width=True):
            if not target:
                st.error("请选择要删除的表")
            elif confirm_name.strip() != target:
                st.error("确认表名不匹配")
            else:
                with st.spinner("正在删除表格..."):
                    result = delete_table_api(target)
                if result.get("success"):
                    st.success(f"删除成功：{result.get('table_name')}")
                    st.cache_data.clear()
                    if st.session_state.get("selected_table") == target:
                        st.session_state.selected_table = None
                    if target in st.session_state.get("selected_tables", []):
                        st.session_state.selected_tables = [t for t in st.session_state.get("selected_tables", []) if t != target]
                    st.rerun()
                else:
                    st.error(result.get("error") or "删除失败")
    else:
        st.info("当前没有可删除的数据表")

st.markdown("<br><hr><div style='text-align: center; color: #999; font-size: 0.8rem;'>NL2SQL Management v1.2</div>", unsafe_allow_html=True)
