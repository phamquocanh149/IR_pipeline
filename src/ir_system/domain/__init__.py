"""Domain package — core value objects."""

from ir_system.domain.document import Document
from ir_system.domain.hit import Hit
from ir_system.domain.qrels import Qrels
from ir_system.domain.query import Query
from ir_system.domain.run import Run

__all__ = ["Query", "Document", "Hit", "Qrels", "Run"]
