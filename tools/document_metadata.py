import json
from typing import List

from langchain_core.tools import StructuredTool

from db.file_manager import list_document_metadata, list_uploaded_files, query_document_facts


FACT_TYPE_TERMS = {
    "amount": ("金额", "价格", "总价", "费用", "税额", "多少钱", "amount", "price", "cost"),
    "percentage": ("比例", "百分比", "税率", "利率", "折扣", "percent", "rate"),
    "date": ("日期", "时间", "何时", "哪天", "date", "when"),
    "person": ("人员", "姓名", "签字人", "联系人", "person", "who"),
    "organization": ("公司", "机构", "甲方", "乙方", "organization", "company"),
    "contract_number": ("合同号", "合同编号", "contract number"),
    "invoice_number": ("发票号", "发票编号", "invoice number"),
    "project_number": ("项目号", "项目编号", "project number"),
    "phone": ("电话", "手机号", "phone"),
    "email": ("邮箱", "邮件地址", "email"),
    "payment_term": ("付款条件", "付款方式", "账期", "payment term"),
    "signature": ("签署", "签字", "盖章", "signature"),
    "quantity": ("数量", "多少个", "quantity"),
    "metric": ("指标", "metric", "kpi"),
}


def infer_fact_types(query: str) -> List[str]:
    lowered = query.casefold()
    return [fact_type for fact_type, terms in FACT_TYPE_TERMS.items() if any(term in lowered for term in terms)]


def query_session_document_metadata(
    session_id: str, query: str, file_id: str = "", fact_types: str = "",
) -> str:
    requested_types = [item.strip() for item in fact_types.split(",") if item.strip()] or infer_fact_types(query)
    uploads = {item["file_id"]: item["original_name"] for item in list_uploaded_files(session_id)}
    metadata_rows = list_document_metadata(session_id)
    if file_id:
        metadata_rows = [item for item in metadata_rows if item["file_id"] == file_id]
    facts = query_document_facts(
        session_id, file_id=file_id or None, fact_types=requested_types or None, limit=40,
    )
    for fact in facts:
        fact["source"] = uploads.get(fact["file_id"], fact["file_id"])
    payload = {
        "query": query,
        "requested_fact_types": requested_types,
        "documents": [
            {
                "file_id": row["file_id"], "source": uploads.get(row["file_id"], row["file_id"]),
                "extraction_status": row["extraction_status"], "structure_version": row["structure_version"],
                "metadata": row["metadata"],
            }
            for row in metadata_rows
        ],
        "facts": facts,
        "instructions": (
            "Use only explicit values returned here. Cite the source, physical_page/printed_page and evidence_text. "
            "If multiple values conflict, list them instead of choosing one."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def create_document_metadata_tool(session_id: str):
    def query_document_metadata(query: str, file_id: str = "", fact_types: str = "") -> str:
        """Query structured uploaded-document metadata and exact facts before semantic retrieval."""
        return query_session_document_metadata(session_id, query, file_id, fact_types)

    return StructuredTool.from_function(
        func=query_document_metadata,
        name="query_document_metadata",
        description=(
            "Query exact facts and document properties such as page count, title, author, amounts, currencies, "
            "dates, parties, identifiers, contacts, quantities, signatures, and payment terms. fact_types is an "
            "optional comma-separated list. Always preserve source/page/evidence in the answer."
        ),
    )
