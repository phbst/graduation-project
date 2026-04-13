from io import BytesIO
import json
import os
import random
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# ========== 配置 ==========
DB_PATH = "./funds_v3.db"
PROMPT_TEMPLATE_FILE = "infer.template"
CHART_PROMPT_TEMPLATE_FILE = "chart_infer.template"
DB_CONFIG_FILE = "config.json"
MODEL_CONFIG_FILE = "model_config.json"

TEMPERATURE = 0
REQUEST_TIMEOUT = 30
LOG_FILE = "query_logs.jsonl"
CHART_MODEL_CANDIDATES = [
    "claude-sonnet-4.5",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-6",
]

# ========== 全局变量 ==========
db_config = None
model_config = None

# ========== FastAPI 应用 ==========
app = FastAPI(title="SQL查询API", description="支持多表选择问答的SQL查询服务")


# ========== 请求模型 ==========
class QueryRequest(BaseModel):
    query: str
    table_name: Optional[str] = None
    table_names: Optional[List[str]] = None
    model_name: Optional[str] = None


class RawSQLRequest(BaseModel):
    sql: str


class ChartSuggestionRequest(BaseModel):
    query: str
    sql: Optional[str] = None
    data: List[Dict[str, Any]]
    columns: Optional[List[str]] = None
    model_name: Optional[str] = None


# ========== 响应模型 ==========
class QueryResponse(BaseModel):
    success: bool
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None
    total_rows: Optional[int] = None
    error: Optional[str] = None
    model_response: Optional[str] = None


class TablesResponse(BaseModel):
    success: bool
    tables: List[str]
    count: int


class ModelsResponse(BaseModel):
    success: bool
    models: Dict[str, Dict[str, Any]]
    default_model: Optional[str] = None


class RawSQLResponse(BaseModel):
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None
    total_rows: Optional[int] = None
    error: Optional[str] = None


class ImportExcelResponse(BaseModel):
    success: bool
    table_name: Optional[str] = None
    columns: Optional[List[str]] = None
    total_rows: Optional[int] = None
    build_statement: Optional[str] = None
    error: Optional[str] = None


class DeleteTableResponse(BaseModel):
    success: bool
    table_name: Optional[str] = None
    error: Optional[str] = None


class ChartSuggestionResponse(BaseModel):
    success: bool
    should_visualize: bool
    chart: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None


# ========== 工具函数 ==========
def load_db_config() -> bool:
    global db_config
    try:
        with open(DB_CONFIG_FILE, "r", encoding="utf-8") as f:
            db_config = json.load(f)
        print(f"[INFO] 成功加载数据库配置，包含 {len(db_config)} 个表")
        return True
    except Exception as e:
        print(f"[ERROR] 加载数据库配置失败: {e}")
        return False


def load_model_config() -> bool:
    global model_config
    try:
        with open(MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        print(f"[INFO] 成功加载模型配置，包含 {len(model_config['models'])} 个模型")
        return True
    except Exception as e:
        print(f"[ERROR] 加载模型配置失败: {e}")
        return False


def save_db_config() -> None:
    with open(DB_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(db_config or {}, f, ensure_ascii=False, indent=2)


def save_query_log(record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_model_name(preferred_name: Optional[str] = None, candidates: Optional[List[str]] = None) -> str:
    if not model_config:
        raise HTTPException(status_code=500, detail="模型配置未加载")

    if preferred_name:
        if preferred_name not in model_config["models"]:
            raise HTTPException(status_code=400, detail=f"模型 '{preferred_name}' 不存在")
        return preferred_name

    if candidates:
        for candidate in candidates:
            if candidate in model_config["models"]:
                return candidate

    return model_config.get("default_model", "SFT-Qwen3-8B")


def render_prompt(template_file: str, **kwargs: Any) -> str:
    try:
        with open(template_file, encoding="utf-8") as f:
            return f.read().format(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取prompt模板失败: {e}")


def call_model_with_prompt(prompt: str, model_name: Optional[str] = None, candidates: Optional[List[str]] = None) -> Dict[str, Any]:
    selected_model_name = resolve_model_name(model_name, candidates)
    model_info = model_config["models"][selected_model_name]

    if not model_info.get("enabled", True):
        raise HTTPException(status_code=400, detail=f"模型 '{selected_model_name}' 已禁用")

    if model_info["type"] == "local":
        api_url = f"{model_info['url']}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
        }
    else:
        api_url = f"{model_info['url']}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": model_info["api_key"],
        }
        payload = {
            "model": model_info["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
        }

    try:
        start = time.time()
        resp = requests.post(api_url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        latency = time.time() - start
        print(f"[INFO] 模型 {selected_model_name} 响应耗时: {latency:.2f}s")
        content = data["choices"][0]["message"]["content"]
        return {"model_name": selected_model_name, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用模型 {selected_model_name} 失败: {e}")


def normalize_column_name(col_name: str) -> str:
    name = str(col_name)
    for old, new in [
        ("(", "_"), (")", "_"), ("（", "_"), ("）", "_"), ("\n", "_"),
        ("/", "_"), (".", ""), (" ", "_"), ("-", "_"), (":", "_"),
        ("*", "_"), ("'", "_"), ("+", "_"), ("&", "_"), ("!", "_"), ("×", "_"),
    ]:
        name = name.replace(old, new)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_") or "column"


def normalize_table_name(table_name: str) -> str:
    normalized = re.sub(r"\W+", "_", str(table_name).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise HTTPException(status_code=400, detail="表名不能为空")
    if normalized[0].isdigit():
        normalized = f"table_{normalized}"
    return normalized


def get_sqlite_type_from_series(series: pd.Series) -> str:
    dtype = str(series.dtype)
    if "int" in dtype:
        return "INT"
    if "float" in dtype:
        return "REAL"
    if "bool" in dtype:
        return "BOOLEAN"
    if "datetime" in dtype:
        return "TEXT"
    return "TEXT"


def generate_create_table_with_comments(df: pd.DataFrame, table_name: str) -> str:
    fields = []
    for col in df.columns:
        col_type = get_sqlite_type_from_series(df[col])
        non_nulls = df[col].dropna()
        example_value = ""
        if not non_nulls.empty:
            example_value = str(random.choice(non_nulls.tolist()))
            example_value = example_value.replace("\n", " ").replace("'", "‘").replace('"', "“")
        comment = ""
        if example_value:
            preview = f"{example_value[:10]}..." if len(example_value) > 20 else example_value
            comment = f" COMMENT '样例：{preview}'"
        fields.append(f"`{col}` {col_type}{comment}")
    return f"CREATE TABLE {table_name} ({', '.join(fields)});"


def read_uploaded_table(file_name: str, content: bytes, sheet_name: Optional[str]) -> pd.DataFrame:
    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
        return pd.read_csv(BytesIO(content))
    if lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
        target_sheet = sheet_name if sheet_name else 0
        return pd.read_excel(BytesIO(content), sheet_name=target_sheet)
    raise HTTPException(status_code=400, detail="仅支持上传 .xlsx、.xls 或 .csv 文件")


def import_dataframe_to_db(df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> Dict[str, Any]:
    global db_config

    if if_exists not in {"fail", "replace", "append"}:
        raise HTTPException(status_code=400, detail="if_exists 仅支持 fail、replace、append")

    normalized_columns = [normalize_column_name(col) for col in df.columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise HTTPException(status_code=400, detail="标准化后的列名存在重复，请调整原始表头")

    df = df.copy()
    df.columns = normalized_columns
    normalized_table_name = normalize_table_name(table_name)

    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql(normalized_table_name, conn, if_exists=if_exists, index=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导入数据失败: {e}")
    finally:
        conn.close()

    build_statement = generate_create_table_with_comments(df, normalized_table_name)
    if db_config is None:
        db_config = {}
    db_config[normalized_table_name] = {"build": build_statement}
    save_db_config()

    return {
        "table_name": normalized_table_name,
        "columns": list(df.columns),
        "total_rows": len(df),
        "build_statement": build_statement,
    }


def extract_json_payload(resp: str) -> Dict[str, Any]:
    if not resp:
        raise HTTPException(status_code=500, detail="模型返回为空")

    text = str(resp).strip()
    match = re.search(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    try:
        return json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析模型JSON失败: {e}")


def suggest_chart(query: str, sql: Optional[str], data: List[Dict[str, Any]], columns: Optional[List[str]], model_name: Optional[str]) -> Dict[str, Any]:
    if not data:
        return {
            "should_visualize": False,
            "chart": None,
            "model_name": None,
            "raw_response": None,
        }

    preview_rows = data[:20]
    prompt = render_prompt(
        CHART_PROMPT_TEMPLATE_FILE,
        query=query,
        sql=sql or "",
        columns=json.dumps(columns or list(preview_rows[0].keys()), ensure_ascii=False),
        row_count=len(data),
        preview_data=json.dumps(preview_rows, ensure_ascii=False, indent=2),
    )
    model_result = call_model_with_prompt(prompt, model_name=model_name, candidates=CHART_MODEL_CANDIDATES)
    payload = extract_json_payload(model_result["content"])

    should_visualize = bool(payload.get("should_visualize", False))
    chart = payload.get("chart") if should_visualize else None

    if chart:
        chart.setdefault("title", query)
        chart.setdefault("chart_type", "bar")

    return {
        "should_visualize": should_visualize,
        "chart": chart,
        "model_name": model_result["model_name"],
        "raw_response": model_result["content"],
    }


def delete_table_from_db(table_name: str) -> str:
    global db_config

    normalized_table_name = normalize_table_name(table_name)
    if not db_config or normalized_table_name not in db_config:
        raise HTTPException(status_code=404, detail=f"表 '{normalized_table_name}' 不存在")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(f'DROP TABLE IF EXISTS "{normalized_table_name}"')
        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除表失败: {e}")
    finally:
        conn.close()

    db_config.pop(normalized_table_name, None)
    save_db_config()
    return normalized_table_name


def call_model_api(query: str, table_names: Optional[List[str]] = None, model_name: Optional[str] = None) -> str:
    build_statement = ""
    if table_names and db_config:
        table_builds = []
        for table_name in table_names:
            if table_name in db_config:
                table_builds.append(f"【{table_name}】\n{db_config[table_name]['build']}")
        build_statement = "\n\n".join(table_builds)

    prompt = render_prompt(PROMPT_TEMPLATE_FILE, query=query, build=build_statement)
    return call_model_with_prompt(prompt, model_name=model_name)["content"]


def extract_sql(resp: str) -> str:
    if not resp:
        return resp
    text = str(resp).strip()
    pattern = r"```(?:sql)?\s*([\s\S]*?)```"
    m = re.match(pattern, text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def fix_table_name(sql: str, table_names: Optional[List[str]] = None) -> str:
    return sql


def execute_sql(sql: str) -> Dict[str, Any]:
    if not sql:
        raise HTTPException(status_code=400, detail="SQL语句为空")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(sql)
        if sql.strip().upper().startswith("SELECT"):
            rows = cur.fetchall()
            columns = [c[0] for c in cur.description]
            data = [{columns[i]: row[i] for i in range(len(columns))} for row in rows]
            return {"data": data, "columns": columns, "total_rows": len(data)}
        conn.commit()
        return {"data": [], "columns": [], "total_rows": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL执行失败: {e}")
    finally:
        conn.close()


def preview_table(table_name: str, limit: int = 10) -> Dict[str, Any]:
    if not db_config:
        raise HTTPException(status_code=500, detail="数据库配置未加载")
    if table_name not in db_config:
        raise HTTPException(status_code=404, detail=f"表 '{table_name}' 不存在")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(f'SELECT * FROM "{table_name}" LIMIT ?', (limit,))
        rows = cur.fetchall()
        columns = [c[0] for c in cur.description] if cur.description else []
        data = [{columns[i]: row[i] for i in range(len(columns))} for row in rows]
        return {"data": data, "columns": columns, "total_rows": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"表预览失败: {e}")
    finally:
        conn.close()


# ========== API路由 ==========
@app.on_event("startup")
async def startup_event():
    if not load_db_config():
        raise RuntimeError("无法加载数据库配置")
    if not load_model_config():
        raise RuntimeError("无法加载模型配置")


@app.get("/", summary="根路径")
async def root():
    return {"message": "SQL查询API服务运行中", "version": "1.0.0"}


@app.get("/tables", response_model=TablesResponse, summary="获取可用表列表")
async def get_tables():
    if not db_config:
        raise HTTPException(status_code=500, detail="数据库配置未加载")
    tables = list(db_config.keys())
    return TablesResponse(success=True, tables=tables, count=len(tables))


@app.get("/tables/{table_name}/schema", summary="获取表结构")
async def get_table_schema(table_name: str):
    if not db_config:
        raise HTTPException(status_code=500, detail="数据库配置未加载")
    if table_name not in db_config:
        raise HTTPException(status_code=404, detail=f"表 '{table_name}' 不存在")
    return {"table_name": table_name, "build_statement": db_config[table_name]["build"]}


@app.get("/table_preview/{table_name}", response_model=RawSQLResponse, summary="预览数据表")
async def get_table_preview(table_name: str, limit: int = 10):
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit 必须大于 0")
    result = preview_table(table_name, limit)
    return RawSQLResponse(success=True, data=result["data"], columns=result["columns"], total_rows=result["total_rows"])


@app.get("/models", response_model=ModelsResponse, summary="获取可用模型列表")
async def get_models():
    if not model_config:
        raise HTTPException(status_code=500, detail="模型配置未加载")
    enabled_models = {name: info for name, info in model_config["models"].items() if info.get("enabled", True)}
    return ModelsResponse(success=True, models=enabled_models, default_model=model_config.get("default_model"))


@app.post("/query", response_model=QueryResponse, summary="执行SQL查询")
async def query_data(request: QueryRequest):
    if not db_config:
        raise HTTPException(status_code=500, detail="数据库配置未加载")

    table_names: List[str] = []
    if request.table_names:
        table_names = request.table_names
        for table_name in table_names:
            if table_name not in db_config:
                raise HTTPException(status_code=400, detail=f"表 '{table_name}' 不存在")
    elif request.table_name:
        if request.table_name not in db_config:
            raise HTTPException(status_code=400, detail=f"表 '{request.table_name}' 不存在")
        table_names = [request.table_name]
    else:
        raise HTTPException(status_code=400, detail="必须指定table_name或table_names")

    query_text = request.query
    model_name = request.model_name
    model_response = ""
    sql = ""
    log_type = 0

    try:
        model_response = call_model_api(query_text, table_names, model_name)
        print(f"[INFO] 模型请求成功 {model_response}")
        sql = extract_sql(model_response)
        print(f"[INFO] 提取SQL成功 {sql}")
        if sql == model_response.strip():
            save_query_log({"query": query_text, "tables": table_names, "llm_res": model_response, "sql": "", "type": 0})
            return QueryResponse(success=False, error="无法从模型响应中提取SQL语句", model_response=model_response)

        sql = fix_table_name(sql, table_names)

        try:
            result = execute_sql(sql)
            total_rows = result.get("total_rows", 0)
            log_type = 3 if total_rows > 0 else 2
        except Exception:
            log_type = 1
            raise

        save_query_log({"query": query_text, "tables": table_names, "llm_res": model_response, "sql": sql, "type": log_type})
        return QueryResponse(
            success=True,
            sql=sql,
            data=result["data"],
            columns=result["columns"],
            total_rows=result["total_rows"],
            model_response=model_response,
        )
    except Exception as e:
        if log_type == 0:
            save_query_log({"query": query_text, "tables": table_names, "llm_res": model_response, "sql": sql, "type": 1 if sql else 0})
        return QueryResponse(success=False, error=str(e))


@app.post("/execute_raw_sql", response_model=RawSQLResponse, summary="执行原生SQL")
async def execute_raw_sql_endpoint(request: RawSQLRequest):
    sql = request.sql.strip() if request.sql else ""
    if not sql:
        return RawSQLResponse(success=False, error="SQL语句为空")

    try:
        result = execute_sql(sql)
        return RawSQLResponse(success=True, data=result["data"], columns=result["columns"], total_rows=result["total_rows"])
    except Exception as e:
        return RawSQLResponse(success=False, error=str(e))


@app.post("/import_excel", response_model=ImportExcelResponse, summary="导入表格到数据库")
async def import_excel(
    file: UploadFile = File(...),
    table_name: str = Form(...),
    sheet_name: Optional[str] = Form(None),
    if_exists: str = Form("replace"),
):
    try:
        content = await file.read()
        if not content:
            return ImportExcelResponse(success=False, error="上传文件为空")

        df = read_uploaded_table(file.filename or "uploaded.xlsx", content, sheet_name)
        result = import_dataframe_to_db(df, table_name, if_exists)
        return ImportExcelResponse(
            success=True,
            table_name=result["table_name"],
            columns=result["columns"],
            total_rows=result["total_rows"],
            build_statement=result["build_statement"],
        )
    except HTTPException as e:
        return ImportExcelResponse(success=False, error=str(e.detail))
    except Exception as e:
        return ImportExcelResponse(success=False, error=str(e))


@app.delete("/tables/{table_name}", response_model=DeleteTableResponse, summary="删除数据表")
async def delete_table(table_name: str):
    try:
        deleted_table_name = delete_table_from_db(table_name)
        return DeleteTableResponse(success=True, table_name=deleted_table_name)
    except HTTPException as e:
        return DeleteTableResponse(success=False, error=str(e.detail))
    except Exception as e:
        return DeleteTableResponse(success=False, error=str(e))


@app.post("/chart_suggestion", response_model=ChartSuggestionResponse, summary="根据SQL结果建议可视化图表")
async def chart_suggestion(request: ChartSuggestionRequest):
    try:
        result = suggest_chart(
            query=request.query,
            sql=request.sql,
            data=request.data,
            columns=request.columns,
            model_name=request.model_name,
        )
        return ChartSuggestionResponse(
            success=True,
            should_visualize=result["should_visualize"],
            chart=result["chart"],
            model_name=result["model_name"],
            raw_response=result["raw_response"],
        )
    except HTTPException as e:
        return ChartSuggestionResponse(
            success=False,
            should_visualize=False,
            chart=None,
            error=str(e.detail),
        )
    except Exception as e:
        return ChartSuggestionResponse(
            success=False,
            should_visualize=False,
            chart=None,
            error=str(e),
        )


@app.get("/health", summary="健康检查")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "tables_loaded": len(db_config) if db_config else 0,
        "models_loaded": len(model_config["models"]) if model_config else 0,
        "default_model": model_config.get("default_model") if model_config else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8999)
