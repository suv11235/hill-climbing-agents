from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DocumentType(str, Enum):
    INVOICE = "invoice"
    FORM = "form"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    required: bool = True


@dataclass(frozen=True)
class DocumentSchema:
    doc_type: DocumentType
    fields: tuple[FieldSpec, ...]

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]


INVOICE_SCHEMA = DocumentSchema(
    doc_type=DocumentType.INVOICE,
    fields=(
        FieldSpec("invoice_number", "Invoice #"),
        FieldSpec("date", "Date"),
        FieldSpec("vendor", "Vendor"),
        FieldSpec("total", "Total"),
        FieldSpec("tax", "Tax"),
    ),
)

FORM_SCHEMA = DocumentSchema(
    doc_type=DocumentType.FORM,
    fields=(
        FieldSpec("name", "Name"),
        FieldSpec("email", "Email"),
        FieldSpec("phone", "Phone"),
        FieldSpec("id_number", "ID Number"),
    ),
)

SCHEMAS: dict[DocumentType, DocumentSchema] = {
    DocumentType.INVOICE: INVOICE_SCHEMA,
    DocumentType.FORM: FORM_SCHEMA,
}
