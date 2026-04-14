import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import HTTPException

try:
    import jieba
except ImportError:
    jieba = None

try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None


ES_HOST = os.getenv("ES_HOST", "127.0.0.1")
ES_PORT = int(os.getenv("ES_PORT", "9200"))
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "nl2sql_schema_index")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://api.vectorengine.ai/v1/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
REQUEST_TIMEOUT = int(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "30"))
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("VECTORENGINE_API_KEY", ""))
PROMPTBUILD_MAX_VALUES_PER_COLUMN = int(os.getenv("PROMPTBUILD_MAX_VALUES_PER_COLUMN", "3"))


class ESPromptBuilder:
    def __init__(self) -> None:
        self.es_client = None

    def init_es_client(self) -> bool:
        if Elasticsearch is None:
            print("[WARN] elasticsearch 未安装，/promptbuild 接口将不可用")
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

        if result.get("data"):
            return [item.get("embedding") for item in result["data"]]

        raise HTTPException(status_code=500, detail=f"embedding 接口返回异常: {result}")

    def _embed_query_segments(self, segments: List[str]) -> List[List[float]]:
        headers = self._generate_authorization_header()
        vectors: List[List[float]] = []

        for i in range(0, len(segments), EMBEDDING_BATCH_SIZE):
            batch = segments[i:i + EMBEDDING_BATCH_SIZE]
            embeddings = self._get_embeddings(batch, headers)
            for embedding in embeddings:
                if embedding:
                    vectors.append(embedding)
        return vectors

    def _tokenize_query(self, query: str) -> List[str]:
        query = (query or "").strip()
        if not query:
            return []

        segments: List[str] = []
        if jieba is not None:
            segments.extend([word.strip() for word in jieba.cut(query, cut_all=False) if word.strip()])
        else:
            segments.extend([part.strip() for part in re.split(r"[\s,，。！？；、]+", query) if part.strip()])

        if query not in segments:
            segments.append(query)

        unique_segments: List[str] = []
        seen = set()
        for item in segments:
            if item not in seen:
                unique_segments.append(item)
                seen.add(item)
        return unique_segments

    def _split_sql_fields(self, fields_text: str) -> List[str]:
        fields: List[str] = []
        current: List[str] = []
        in_quote = False
        quote_char = ""

        for char in fields_text:
            if char in {"'", '"'}:
                if not in_quote:
                    in_quote = True
                    quote_char = char
                elif quote_char == char:
                    in_quote = False
                    quote_char = ""
            if char == "," and not in_quote:
                item = "".join(current).strip()
                if item:
                    fields.append(item)
                current = []
                continue
            current.append(char)

        tail = "".join(current).strip()
        if tail:
            fields.append(tail)
        return fields

    def _parse_build_statement(self, build_statement: str) -> List[Dict[str, str]]:
        if not build_statement:
            return []

        start = build_statement.find("(")
        end = build_statement.rfind(")")
        if start == -1 or end == -1 or end <= start:
            return []

        fields_text = build_statement[start + 1:end]
        parsed_columns: List[Dict[str, str]] = []

        for raw_field in self._split_sql_fields(fields_text):
            field = raw_field.strip()
            if not field.startswith("`"):
                continue

            name_end = field.find("`", 1)
            if name_end == -1:
                continue

            column_name = field[1:name_end]
            remainder = field[name_end + 1:].strip()
            if not remainder:
                continue

            type_part, _, comment_part = remainder.partition(" COMMENT ")
            column_type = type_part.strip()
            comment = comment_part.strip()
            if comment.startswith("'") and comment.endswith("'") and len(comment) >= 2:
                comment = comment[1:-1]

            parsed_columns.append({
                "name": column_name,
                "type": column_type,
                "comment": comment,
            })

        return parsed_columns

    def _ensure_es_ready(self) -> None:
        if self.es_client is None:
            raise HTTPException(status_code=500, detail="Elasticsearch 未连接，无法构建压缩 prompt")

    def _search_schema_hits(
        self,
        query_vector: List[float],
        table_names: List[str],
        search_type: str,
        size: int,
    ) -> List[Dict[str, Any]]:
        self._ensure_es_ready()

        search_body = {
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {"table_name": table_names}},
                        {"term": {"type": search_type}},
                    ],
                    "must": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                                "params": {"query_vector": query_vector},
                            },
                        }
                    },
                }
            },
            "_source": ["text", "table_name", "column_name", "column_names", "type"],
            "size": size,
        }

        try:
            response = self.es_client.search(index=ES_INDEX_NAME, body=search_body)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ES 检索失败: {e}")

        hits: List[Dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            column_names = source.get("column_names") or [source.get("column_name", "")]
            for column_name in column_names:
                hits.append({
                    "text": source.get("text", ""),
                    "table_name": source.get("table_name", ""),
                    "column_name": column_name,
                    "type": source.get("type", ""),
                    "score": round(float(hit.get("_score", 0.0)) - 1.0, 4),
                })
        return hits

    def _deduplicate_preserve_order(self, items: List[str], limit: Optional[int] = None) -> List[str]:
        result: List[str] = []
        seen = set()
        for item in items:
            value = str(item).strip()
            if not value or value in seen:
                continue
            result.append(value)
            seen.add(value)
            if limit is not None and len(result) >= limit:
                break
        return result

    def recall_columns(
        self,
        query: str,
        table_names: List[str],
        topk: int,
    ) -> List[Dict[str, Any]]:
        self._ensure_es_ready()

        if topk <= 0:
            raise HTTPException(status_code=400, detail="topk 必须大于 0")

        segments = self._tokenize_query(query)
        if not segments:
            raise HTTPException(status_code=400, detail="query 不能为空")

        vectors = self._embed_query_segments(segments)
        if not vectors:
            raise HTTPException(status_code=500, detail="query 向量化失败")

        recall_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        search_size = max(topk * 3, 10)

        for vector in vectors:
            for hit in self._search_schema_hits(vector, table_names, "column_name", search_size):
                key = (hit["table_name"], hit["column_name"])
                item = recall_map.setdefault(key, {
                    "table_name": hit["table_name"],
                    "column_name": hit["column_name"],
                    "column_similarity": -1.0,
                    "value_similarity": -1.0,
                    "best_similarity": -1.0,
                    "matched_value": "",
                    "matched_text": hit["column_name"],
                    "match_type": "column_name",
                })
                if hit["score"] > item["column_similarity"]:
                    item["column_similarity"] = hit["score"]
                if hit["score"] > item["best_similarity"]:
                    item["best_similarity"] = hit["score"]
                    item["matched_text"] = hit["column_name"]
                    item["match_type"] = "column_name"

            for hit in self._search_schema_hits(vector, table_names, "column_value", search_size):
                key = (hit["table_name"], hit["column_name"])
                item = recall_map.setdefault(key, {
                    "table_name": hit["table_name"],
                    "column_name": hit["column_name"],
                    "column_similarity": -1.0,
                    "value_similarity": -1.0,
                    "best_similarity": -1.0,
                    "matched_value": "",
                    "matched_text": hit["text"],
                    "match_type": "column_value",
                })
                if hit["score"] > item["value_similarity"]:
                    item["value_similarity"] = hit["score"]
                if hit["score"] > item["best_similarity"]:
                    item["best_similarity"] = hit["score"]
                    item["matched_value"] = hit["text"]
                    item["matched_text"] = hit["text"]
                    item["match_type"] = "column_value"

        ranked_hits = sorted(
            recall_map.values(),
            key=lambda item: item["best_similarity"],
            reverse=True,
        )
        return ranked_hits[:topk]

    def build_prompt_schema(
        self,
        query: str,
        table_names: List[str],
        db_config: Dict[str, Any],
        column_topk: int,
        value_topk: int,
        similarity_threshold: float,
        max_columns_per_table: int,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not db_config:
            raise HTTPException(status_code=500, detail="数据库配置未加载")

        missing_tables = [table_name for table_name in table_names if table_name not in db_config]
        if missing_tables:
            raise HTTPException(status_code=400, detail=f"表不存在: {', '.join(missing_tables)}")

        segments = self._tokenize_query(query)
        if not segments:
            raise HTTPException(status_code=400, detail="query 不能为空")

        vectors = self._embed_query_segments(segments)
        if not vectors:
            raise HTTPException(status_code=500, detail="query 向量化失败")

        candidates: Dict[Tuple[str, str], Dict[str, Any]] = {}
        raw_value_hits: Dict[Tuple[str, str], List[str]] = {}

        for vector in vectors:
            for hit in self._search_schema_hits(vector, table_names, "column_name", max(column_topk * 3, 10)):
                key = (hit["table_name"], hit["column_name"])
                item = candidates.setdefault(key, {
                    "table_name": hit["table_name"],
                    "column_name": hit["column_name"],
                    "column_score": -1.0,
                    "value_score": -1.0,
                    "matched_values": [],
                })
                item["column_score"] = max(item["column_score"], hit["score"])

            for hit in self._search_schema_hits(vector, table_names, "column_value", max(value_topk * 4, 10)):
                key = (hit["table_name"], hit["column_name"])
                item = candidates.setdefault(key, {
                    "table_name": hit["table_name"],
                    "column_name": hit["column_name"],
                    "column_score": -1.0,
                    "value_score": -1.0,
                    "matched_values": [],
                })
                item["value_score"] = max(item["value_score"], hit["score"])
                raw_value_hits.setdefault(key, []).append(hit["text"])

        table_to_candidates: Dict[str, List[Dict[str, Any]]] = {table_name: [] for table_name in table_names}
        for key, item in candidates.items():
            item["matched_values"] = self._deduplicate_preserve_order(
                raw_value_hits.get(key, []),
                limit=PROMPTBUILD_MAX_VALUES_PER_COLUMN,
            )
            item["best_score"] = max(item["column_score"], item["value_score"])
            table_to_candidates[item["table_name"]].append(item)

        simplified_statements: List[str] = []
        matched_schema: List[Dict[str, Any]] = []

        for table_name in table_names:
            parsed_columns = self._parse_build_statement(db_config[table_name]["build"])
            if not parsed_columns:
                simplified_statements.append(f"【{table_name}】\n{db_config[table_name]['build']}")
                matched_schema.append({
                    "table_name": table_name,
                    "selected_columns": [],
                    "fallback_full_schema": True,
                })
                continue

            candidate_items = sorted(
                table_to_candidates.get(table_name, []),
                key=lambda item: item["best_score"],
                reverse=True,
            )

            selected_names = [
                item["column_name"]
                for item in candidate_items
                if item["best_score"] >= similarity_threshold
            ][:max_columns_per_table]

            if not selected_names:
                selected_names = [item["column_name"] for item in candidate_items[:max_columns_per_table]]

            selected_name_set = set(selected_names)
            selected_columns_meta: List[Dict[str, Any]] = []
            ddl_fields: List[str] = []

            for column in parsed_columns:
                if column["name"] not in selected_name_set:
                    continue

                candidate_key = (table_name, column["name"])
                candidate_meta = candidates.get(candidate_key, {})
                matched_values = candidate_meta.get("matched_values", [])
                comment_parts = []
                if column["comment"]:
                    comment_parts.append(column["comment"])
                if matched_values:
                    comment_parts.append(f"召回值：{'、'.join(matched_values)}")

                comment_sql = ""
                if comment_parts:
                    comment_text = "；".join(comment_parts).replace("'", "‘")
                    comment_sql = f" COMMENT '{comment_text}'"

                ddl_fields.append(f"`{column['name']}` {column['type']}{comment_sql}")
                selected_columns_meta.append({
                    "column_name": column["name"],
                    "column_type": column["type"],
                    "column_score": candidate_meta.get("column_score"),
                    "value_score": candidate_meta.get("value_score"),
                    "matched_values": matched_values,
                    "original_comment": column["comment"],
                })

            if not ddl_fields:
                simplified_statements.append(f"【{table_name}】\n{db_config[table_name]['build']}")
                matched_schema.append({
                    "table_name": table_name,
                    "selected_columns": [],
                    "fallback_full_schema": True,
                })
                continue

            simplified_sql = f"CREATE TABLE {table_name} ({', '.join(ddl_fields)});"
            simplified_statements.append(f"【{table_name}】\n{simplified_sql}")
            matched_schema.append({
                "table_name": table_name,
                "selected_columns": selected_columns_meta,
                "fallback_full_schema": False,
            })

        return "\n\n".join(simplified_statements), matched_schema


es_prompt_builder = ESPromptBuilder()
