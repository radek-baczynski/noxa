from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from noxa.schemas import SearchResultItem


class SearchProvider(ABC):
    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int,
        region: str | None = None,
        safe_search: str | None = None,
    ) -> list[SearchResultItem]:
        ...
