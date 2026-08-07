from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models.asset import AssetClass
from app.infrastructure.db.models import AssetORM


class AssetRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_or_create(
        self,
        ticker: str,
        name: str | None = None,
        asset_class: AssetClass = AssetClass.EQUITY,
        currency: str = "USD",
    ) -> AssetORM:
        asset = self.db.get(AssetORM, ticker)
        if asset is not None:
            return asset
        asset = AssetORM(ticker=ticker, name=name or ticker, asset_class=asset_class, currency=currency)
        self.db.add(asset)
        self.db.commit()
        self.db.refresh(asset)
        return asset

    def list_all(self) -> list[AssetORM]:
        return list(self.db.scalars(select(AssetORM)).all())
