"""
RAG 检索增强生成服务

基于 ChromaDB 和 Sentence-Transformers/TF-IDF，实现文档加载、分块、向量化和检索功能。
初始知识库使用 skills/巴菲特投资思维/references/ 下的8个Markdown文件。
支持两种嵌入方案：
1. Sentence-Transformers（优先，需网络下载模型）
2. TF-IDF（轻量级fallback，支持中文分词）
"""

import os
import sys
import threading
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

from app.config import settings

_rag_lock = threading.Lock()
_rag_service_instance: Optional["RAGService"] = None


def _chinese_tokenizer(text: str) -> List[str]:
    tokens = []
    text = text.strip()

    try:
        import jieba
        words = jieba.lcut(text)
        for word in words:
            word = word.strip()
            if word and len(word) >= 1:
                tokens.append(word)
        return tokens
    except ImportError:
        pass

    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    english_pattern = re.compile(r'[a-zA-Z]+')
    number_pattern = re.compile(r'\d+')

    segments = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue

        if '\u4e00' <= text[pos] <= '\u9fff':
            end = pos
            while end < len(text) and '\u4e00' <= text[end] <= '\u9fff':
                end += 1
            chinese_word = text[pos:end]
            for i in range(len(chinese_word)):
                segments.append(chinese_word[i])
                if i < len(chinese_word) - 1:
                    segments.append(chinese_word[i:i+2])
            pos = end
        elif text[pos].isalpha():
            m = english_pattern.match(text, pos)
            if m:
                segments.append(m.group().lower())
                pos = m.end()
            else:
                pos += 1
        elif text[pos].isdigit():
            m = number_pattern.match(text, pos)
            if m:
                segments.append(m.group())
                pos = m.end()
            else:
                pos += 1
        else:
            pos += 1

    return segments


class TFIDFEmbedding:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            tokenizer=_chinese_tokenizer,
            lowercase=False,
        )
        self._np = np
        self._fitted = False

    def fit(self, documents: List[str]):
        self._vectorizer.fit(documents)
        self._fitted = True

    def encode(self, text: str) -> Any:
        if isinstance(text, str):
            text = [text]
        if not self._fitted:
            self.fit(text)
        return self._np.array(self._vectorizer.transform(text).toarray())[0]


class RAGService:
    def __init__(self):
        self._chroma_client = None
        self._collection = None
        self._embedding_model = None
        self._embedding_type = None
        self._initialized = False
        self._index_built = False

    def _init_chroma(self):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        vector_dir = settings.DATA_DIR / settings.VECTOR_STORE_DIR
        vector_dir.mkdir(parents=True, exist_ok=True)

        self._chroma_client = chromadb.PersistentClient(
            path=str(vector_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name="buffett_investment",
            metadata={"hnsw:space": "cosine"},
        )

    def _init_embedding(self):
        backend = settings.EMBEDDING_BACKEND.lower()

        if backend == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer

                model_path = settings.PROJECT_ROOT / settings.EMBEDDING_MODEL_PATH
                if model_path.exists() and (model_path / "config.json").exists():
                    self._embedding_model = SentenceTransformer(str(model_path))
                    print("[RAG] 从本地路径加载 Sentence-Transformers 模型: {}".format(model_path))
                else:
                    self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                    print("[RAG] 从远程加载 Sentence-Transformers 模型: {}".format(settings.EMBEDDING_MODEL))
                self._embedding_type = "sentence_transformers"
            except Exception as e:
                print("[RAG] Sentence-Transformers 加载失败: {}".format(e))
                print("[RAG] 回退到 TF-IDF 嵌入方案")
                self._embedding_model = TFIDFEmbedding()
                self._embedding_type = "tfidf"
        else:
            self._embedding_model = TFIDFEmbedding()
            self._embedding_type = "tfidf"
            print("[RAG] 使用 TF-IDF 嵌入方案")

    def _load_markdown_files(self, dir_path: Path) -> List[Dict[str, str]]:
        documents = []
        if not dir_path.exists():
            return documents

        for md_file in sorted(dir_path.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                documents.append({
                    "id": md_file.stem,
                    "title": md_file.stem,
                    "content": content,
                    "source": str(md_file),
                })
            except Exception as e:
                print("加载文件失败 {}: {}".format(md_file, e))
        return documents

    def _chunk_text(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        chunks = []
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - chunk_overlap
        return chunks

    def build_index(self, force_rebuild: bool = False):
        if self._index_built and not force_rebuild:
            return

        if not self._initialized:
            self._ensure_init()

        if force_rebuild and self._collection.count() > 0:
            self._chroma_client.delete_collection("buffett_investment")
            self._collection = self._chroma_client.create_collection(
                name="buffett_investment",
                metadata={"hnsw:space": "cosine"},
            )

        if self._collection.count() > 0:
            self._index_built = True
            return

        skills_dir = settings.SKILLS_DIR / "巴菲特投资思维" / "skills" / "buffett" / "references"
        documents = self._load_markdown_files(skills_dir)

        if not documents:
            print("未找到知识库文档，跳过索引构建")
            return

        all_ids = []
        all_embeddings = []
        all_metadatas = []
        all_documents = []

        for doc in documents:
            chunks = self._chunk_text(
                doc["content"],
                settings.RAG_CHUNK_SIZE,
                settings.RAG_CHUNK_OVERLAP,
            )
            for i, chunk in enumerate(chunks):
                doc_id = "{}_{}".format(doc["id"], i)
                all_ids.append(doc_id)
                all_documents.append(chunk)
                all_metadatas.append({
                    "title": doc["title"],
                    "source": doc["source"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                })

        print("正在向量化 {} 个文档块...".format(len(all_documents)))

        if self._embedding_type == "tfidf":
            self._embedding_model.fit(all_documents)

        all_embeddings = [self._embedding_model.encode(doc).tolist() for doc in all_documents]

        self._collection.add(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_documents,
            metadatas=all_metadatas,
        )

        print("索引构建完成，共 {} 个文档块".format(len(all_documents)))
        self._index_built = True

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if not self._initialized:
            self._ensure_init()

        if not self._index_built:
            self.build_index()

        k = top_k or settings.RAG_TOP_K
        query_embedding = self._embedding_model.encode(query).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return matches

    def retrieve_context(self, query: str, top_k: int = None) -> str:
        matches = self.retrieve(query, top_k)
        if not matches:
            return ""

        chunks = []
        for match in matches:
            source = match["metadata"].get("title", "unknown")
            content = match["content"]
            chunks.append("【{}】\n{}\n".format(source, content))

        return "\n".join(chunks)

    def _ensure_init(self):
        if self._initialized:
            return
        with _rag_lock:
            if self._initialized:
                return
            self._init_chroma()
            self._init_embedding()
            self._initialized = True

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._index_built

    @property
    def document_count(self) -> int:
        if not self._collection:
            return 0
        return self._collection.count()

    @property
    def embedding_type(self) -> str:
        return self._embedding_type or "unknown"


def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        with _rag_lock:
            if _rag_service_instance is None:
                _rag_service_instance = RAGService()
    return _rag_service_instance


def initialize_rag():
    rag = get_rag_service()
    rag.build_index()
    return rag
