from typing import Protocol

from app.schemas.portfolio import PortfolioAccountSnapshot


class PortfolioProvider(Protocol):
    def get_account_snapshot(self, account_id: str, tenant_id: str | None = None) -> PortfolioAccountSnapshot | None:
        """Return read-only account snapshot. No trade execution."""
        ...


class InMemoryPortfolioProvider:
    def __init__(self, snapshots: dict[str, PortfolioAccountSnapshot] | None = None) -> None:
        self._snapshots = snapshots or {}

    def get_account_snapshot(self, account_id: str, tenant_id: str | None = None) -> PortfolioAccountSnapshot | None:
        snapshot = self._snapshots.get(account_id)
        if snapshot is None:
            return None
        if tenant_id is not None and snapshot.tenant_id != tenant_id:
            return None
        return snapshot
