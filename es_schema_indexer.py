import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
from fastapi import HTTPException

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
except ImportError:
    Elasticsearch = None
    bulk = None


ES_HOST = os.getenv("ES_HOST", "127.0.0.1")
ES_PORT = int(os.getenv("ES_PORT", "9200"))
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "nl2sql_schema_index")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://api.vectorengine.ai/v1/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1536"))
REQUEST_TIMEOUT = int(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "30"))
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("VECTORENGINE_API_KEY", ""))
MAX_VALUES_PER_COLUMN = int(os.getenv("ES_MAX_VALUES_PER_COLUMN", "1000"))
MIN_VALUE_LENGTH = int(os.getenv("ES_MIN_VALUE_LENGTH", "2"))
MAX_VALUE_LENGTH = int(os.getenv("ES_MAX_VALUE_LENGTH", "50"))
EMBEDDING_CONCURRENCY = int(os.getenv("EMBEDDING_CONCURRENCY", "4"))
EXCLUDE_VALUE_TYPES = ["INTEGER", "REAL", "NUMERIC", "DATE", "DATETIME"]

INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "vector": {
                "type": "dense_vector",
                "dims": VECTOR_DIMENSION,
            },
            "type": {"type": "keyword"},
            "table_name": {"type": "keyword"},
            "column_name": {"type": "keyword"},
            "column_names": {"type": "keyword"},
        }
    },
}


class ESSchemaIndexer:
    def __init__(self) -> None:
        self.es_client = None

    def init_es_client(self) -> bool:
        if Elasticsearch is None:
            print("[WARN] elasticsearch 未安装，ES 索引接口将不可用")
            self.es_client = None
            return False

        try:
            self.es_client = Elasticsearch(hosts=[f"http://{ES_HOST}:{ES_PORT}"])
            self.es_client.info()
            print(f"[INFO] 成功连接 Elasticsearch: http://{ES_HOST}:{ES_PORT}")
            return True
        except Exception as e:
            print(f"[WARN] Elasticsearch 连接失败: {e}")
            self.es_client = None
            return False

    def _ensure_ready(self) -> None:
        if self.es_client is None:
            raise HTTPException(status_code=500, detail="Elasticsearch 未连接，无法执行索引操作")
        if bulk is None:
            raise HTTPException(status_code=500, detail="缺少 elasticsearch 依赖，无法执行索引操作")

    def _generate_authorization_header(self) -> Dict[str, str]:
        if not EMBEDDING_API_KEY:
            raise HTTPException(status_code=500, detail="未配置 EMBEDDING_API_KEY 或 VECTORENGINE_API_KEY")
        return {
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json",
        }

    def _get_embeddings(self, texts: List[str], headers: Dict[str, str]) -> List[Optional[List[float]]]:
        if not texts:
            return []

        payload = {"model": EMBEDDING_MODEL, "input": texts}
        try:
            response = requests.post(
                EMBEDDING_API_URL,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"调用 embedding 接口失败: {e}")

        if not result.get("data"):
            raise HTTPException(status_code=500, detail=f"embedding 接口返回异常: {result}")

        return [item.get("embedding") for item in result["data"]]

    def _get_sqlite_tables(self, conn: sqlite3.Connection) -> List[str]:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [table[0] for table in cursor.fetchall() if not table[0].startswith("sqlite_")]

    def _get_table_schema(self, conn: sqlite3.Connection, table_name: str) -> List[Dict[str, str]]:
        cursor = conn.cursor()
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        return [{"name": col[1], "type": col[2] or "TEXT"} for col in cursor.fetchall()]

    def _create_index(self) -> None:
        self._ensure_ready()
        try:
            if self.es_client.indices.exists(index=ES_INDEX_NAME):
                self.es_client.indices.delete(index=ES_INDEX_NAME, ignore=[400, 404])
            self.es_client.indices.create(index=ES_INDEX_NAME, body=INDEX_MAPPING)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"创建 ES 索引失败: {e}")

    def _ensure_index_exists(self) -> None:
        self._ensure_ready()
        try:
            if not self.es_client.indices.exists(index=ES_INDEX_NAME):
                self.es_client.indices.create(index=ES_INDEX_NAME, body=INDEX_MAPPING)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"检查或创建 ES 索引失败: {e}")

    def _delete_table_docs(self, table_name: str) -> int:
        self._ensure_ready()
        try:
            response = self.es_client.delete_by_query(
                index=ES_INDEX_NAME,
                body={"query": {"term": {"table_name": table_name}}},
                conflicts="proceed",
                refresh=True,
            )
            return int(response.get("deleted", 0))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"删除表 {table_name} 的索引文档失败: {e}")

    def _collect_table_entries(
        self,
        conn: sqlite3.Connection,
        table_name: str,
    ) -> List[Dict[str, Any]]:
        schema = self._get_table_schema(conn, table_name)
        entries: List[Dict[str, Any]] = []
        value_to_columns: Dict[str, set] = {}

        for col_info in schema:
            col_name = col_info["name"]
            col_type = (col_info["type"] or "TEXT").upper()

            entries.append({
                "text": col_name,
                "type": "column_name",
                "table_name": table_name,
                "column_name": col_name,
            })

            if any(exclude_type in col_type for exclude_type in EXCLUDE_VALUE_TYPES):
                continue

            try:
                query = f'SELECT DISTINCT "{col_name}" FROM "{table_name}" LIMIT {MAX_VALUES_PER_COLUMN}'
                values_cursor = conn.cursor()
                values_cursor.execute(query)
                distinct_values = [v[0] for v in values_cursor.fetchall() if v[0] is not None]
            except sqlite3.Error as e:
                print(f"[WARN] 读取 {table_name}.{col_name} 样本值失败: {e}")
                continue

            for val in distinct_values:
                val_str = str(val).strip()
                if MIN_VALUE_LENGTH <= len(val_str) <= MAX_VALUE_LENGTH:
                    value_to_columns.setdefault(val_str, set()).add(col_name)

        for value_text, column_names in value_to_columns.items():
            ordered_column_names = sorted(column_names)
            entries.append({
                "text": value_text,
                "type": "column_value",
                "table_name": table_name,
                "column_name": ordered_column_names[0],
                "column_names": ordered_column_names,
            })

        return entries

    def _entries_to_actions(
        self,
        entries: List[Dict[str, Any]],
        headers: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []

        if not entries:
            return actions

        batches: List[List[Dict[str, Any]]] = [
            entries[i:i + EMBEDDING_BATCH_SIZE]
            for i in range(0, len(entries), EMBEDDING_BATCH_SIZE)
        ]
        ordered_results: List[Optional[List[Dict[str, Any]]]] = [None] * len(batches)

        def process_batch(batch_index: int, batch_entries: List[Dict[str, Any]]) -> tuple:
            batch_texts = [entry["text"] for entry in batch_entries]
            embeddings = self._get_embeddings(batch_texts, headers)
            batch_actions: List[Dict[str, Any]] = []
            for entry, vector in zip(batch_entries, embeddings):
                if vector is None:
                    continue
                source = {
                    "text": entry["text"],
                    "vector": vector,
                    "type": entry["type"],
                    "table_name": entry["table_name"],
                    "column_name": entry["column_name"],
                }
                if entry.get("column_names"):
                    source["column_names"] = entry["column_names"]
                batch_actions.append({
                    "_index": ES_INDEX_NAME,
                    "_source": source,
                })
            return batch_index, batch_actions

        with ThreadPoolExecutor(max_workers=max(1, EMBEDDING_CONCURRENCY)) as executor:
            future_map = {
                executor.submit(process_batch, idx, batch): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(future_map):
                batch_index, batch_actions = future.result()
                ordered_results[batch_index] = batch_actions

        for batch_actions in ordered_results:
            if batch_actions:
                actions.extend(batch_actions)

        return actions

    def _collect_entries_for_tables(self, conn: sqlite3.Connection, table_names: List[str]) -> List[Dict[str, Any]]:
        all_entries: List[Dict[str, Any]] = []
        for table_name in table_names:
            all_entries.extend(self._collect_table_entries(conn, table_name))
        return all_entries

    def rebuild_index(self, db_path: str, table_names: Optional[List[str]] = None) -> Dict[str, Any]:
        self._ensure_ready()
        headers = self._generate_authorization_header()

        try:
            conn = sqlite3.connect(db_path)
        except sqlite3.Error as e:
            raise HTTPException(status_code=500, detail=f"连接 SQLite 失败: {e}")

        try:
            available_tables = self._get_sqlite_tables(conn)
            target_tables = table_names or available_tables
            missing_tables = [name for name in target_tables if name not in available_tables]
            if missing_tables:
                raise HTTPException(status_code=400, detail=f"SQLite 中不存在这些表: {', '.join(missing_tables)}")

            self._create_index()
            all_entries = self._collect_entries_for_tables(conn, target_tables)
            all_actions = self._entries_to_actions(all_entries, headers)

            if all_actions:
                success, errors = bulk(self.es_client, all_actions, index=ES_INDEX_NAME, raise_on_error=False)
                self.es_client.indices.refresh(index=ES_INDEX_NAME)
            else:
                success, errors = 0, []
            doc_count = self.es_client.count(index=ES_INDEX_NAME).get("count", 0)
            return {
                "indexed_tables": target_tables,
                "indexed_documents": int(success),
                "doc_count": int(doc_count),
                "errors": errors[:3] if errors else [],
            }
        finally:
            conn.close()

    def add_tables(self, db_path: str, table_names: List[str]) -> Dict[str, Any]:
        self._ensure_ready()
        if not table_names:
            raise HTTPException(status_code=400, detail="table_names 不能为空")
        headers = self._generate_authorization_header()

        try:
            conn = sqlite3.connect(db_path)
        except sqlite3.Error as e:
            raise HTTPException(status_code=500, detail=f"连接 SQLite 失败: {e}")

        try:
            available_tables = self._get_sqlite_tables(conn)
            missing_tables = [name for name in table_names if name not in available_tables]
            if missing_tables:
                raise HTTPException(status_code=400, detail=f"SQLite 中不存在这些表: {', '.join(missing_tables)}")

            self._ensure_index_exists()

            deleted_docs = 0
            for table_name in table_names:
                deleted_docs += self._delete_table_docs(table_name)

            all_entries = self._collect_entries_for_tables(conn, table_names)
            all_actions = self._entries_to_actions(all_entries, headers)

            if all_actions:
                success, errors = bulk(self.es_client, all_actions, index=ES_INDEX_NAME, raise_on_error=False)
                self.es_client.indices.refresh(index=ES_INDEX_NAME)
            else:
                success, errors = 0, []
            return {
                "indexed_tables": table_names,
                "deleted_documents": deleted_docs,
                "indexed_documents": int(success),
                "errors": errors[:3] if errors else [],
            }
        finally:
            conn.close()

    def remove_tables(self, table_names: List[str]) -> Dict[str, Any]:
        self._ensure_ready()
        if not table_names:
            raise HTTPException(status_code=400, detail="table_names 不能为空")

        try:
            if not self.es_client.indices.exists(index=ES_INDEX_NAME):
                return {
                    "deleted_tables": table_names,
                    "deleted_documents": 0,
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"检查 ES 索引失败: {e}")

        deleted_docs = 0
        for table_name in table_names:
            deleted_docs += self._delete_table_docs(table_name)

        return {
            "deleted_tables": table_names,
            "deleted_documents": deleted_docs,
        }


es_schema_indexer = ESSchemaIndexer()
