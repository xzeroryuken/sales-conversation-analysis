from dataclasses import dataclass, field
from typing import List


@dataclass
class DomainConfig:
    domain: str
    fields: List[str]
    cluster_on: str
    system_prompt: str
    classification_prompt: str
