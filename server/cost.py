import threading
from typing import Tuple

import requests

import config


class CostEstimator:
    def __init__(self, default_per_hour: float):
        self._default_per_hour = default_per_hour
        self._per_hour = default_per_hour
        self._source = "default (not resolved yet)"
        self._lock = threading.Lock()

    @property
    def per_hour(self) -> float:
        with self._lock:
            return self._per_hour

    @property
    def source(self) -> str:
        with self._lock:
            return self._source

    def resolve(self) -> None:
        configured = config.configured_cost_per_hour()
        if configured is not None:
            self._set(configured, "COST_PER_HOUR env")
            return
        try:
            per_hour, source = self._resolve_from_azure()
        except (requests.RequestException, KeyError, ValueError):
            self._set(self._default_per_hour, "default fallback (IMDS/retail API unavailable)")
            return
        self._set(per_hour, source)

    def _resolve_from_azure(self) -> Tuple[float, str]:
        response = requests.get(config.IMDS_METADATA_URL, headers={"Metadata": "true"}, timeout=3)
        response.raise_for_status()
        compute = response.json()["compute"]
        sku = compute["vmSize"]
        region = compute["location"]

        query = (
            f"serviceName eq 'Virtual Machines' and armRegionName eq '{region}' "
            f"and armSkuName eq '{sku}' and priceType eq 'Consumption'"
        )
        response = requests.get(config.RETAIL_PRICES_URL, params={"$filter": query}, timeout=10)
        response.raise_for_status()
        items = [
            item
            for item in response.json()["Items"]
            if "Windows" not in item.get("productName", "") and "Low Priority" not in item.get("skuName", "")
        ]
        if not items:
            raise ValueError("no matching pay-as-you-go retail prices")
        return min(item["unitPrice"] for item in items), f"Azure Retail Prices API ({sku}, {region})"

    def _set(self, per_hour: float, source: str) -> None:
        with self._lock:
            self._per_hour = per_hour
            self._source = source