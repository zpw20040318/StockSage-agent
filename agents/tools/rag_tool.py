"""
智能股票分析助手 — HelloAgents RAG检索工具封装

将RAG检索服务封装为符合HelloAgents标准Tool接口的工具类。
Agent可通过此工具从巴菲特投资思维知识库中检索相关知识。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_HELLO_PATH = _PROJECT_ROOT / "HelloAgents Optimized"
_BACKEND_PATH = _PROJECT_ROOT / "backend"

for p in [_HELLO_PATH, _BACKEND_PATH]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hello_agents.tools import Tool, ToolParameter


class RAGTool(Tool):
    """RAG知识库检索工具

    从巴菲特投资思维知识库中检索与查询相关的知识片段，
    包括思维框架、投资哲学、护城河分析、管理层评估、财务指标、
    估值方法、风险行为、行业策略等内容。

    使用示例:
        tool = RAGTool()
        result = tool.run({"query": "巴菲特护城河分析方法"})
    """

    def __init__(self):
        super().__init__(
            name="rag_search",
            description=(
                "巴菲特投资思维知识库检索工具。从价值投资知识库中检索相关知识，"
                "包括：思维框架、投资哲学、护城河分析、管理层评估、财务指标分析、"
                "估值方法、风险行为分析、行业投资策略等。"
                "用于在分析股票时获取巴菲特投资理念的相关指导原则。"
            ),
        )

        self._rag_service = None

    def _get_rag_service(self):
        if self._rag_service is None:
            from app.services.rag_service import get_rag_service

            self._rag_service = get_rag_service()
            self._rag_service.build_index()
        return self._rag_service

    def get_parameters(self) -> list:
        return [
            ToolParameter(
                name="query",
                type="string",
                description=(
                    "检索查询语句。支持中文查询，例如：\n"
                    "- '巴菲特护城河分析方法'\n"
                    "- '如何评估公司管理层'\n"
                    "- '安全边际计算方法'\n"
                    "- 'ROE财务指标分析'\n"
                    "- '行业竞争格局分析'\n"
                    "- '价值投资估值模型'"
                ),
                required=True,
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="返回结果数量，默认5",
                required=False,
            ),
        ]

    def run(self, parameters: dict) -> str:
        """执行RAG知识库检索

        Args:
            parameters: {"query": "检索查询", "top_k": 5}

        Returns:
            格式化的检索结果文本
        """
        query = parameters.get("query", "")
        if not query:
            return "错误：检索查询内容不能为空"

        top_k = parameters.get("top_k", 5)

        try:
            rag = self._get_rag_service()
            matches = rag.retrieve(query, top_k)

            if not matches:
                return "知识库中未找到相关内容"

            lines = ["## 知识库检索结果"]
            lines.append(f"查询: {query}")
            lines.append("")

            for idx, match in enumerate(matches, 1):
                source = match["metadata"].get("title", "未知来源")
                content = match["content"]
                distance = match.get("distance", 0)

                lines.append(f"### {idx}. {source}")
                lines.append(f"相似度: {1 - distance:.2%}")
                lines.append(content)
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            return f"RAG检索异常: {str(e)}"
