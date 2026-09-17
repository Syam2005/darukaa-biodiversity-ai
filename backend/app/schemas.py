"""Pydantic models describing the API's request and response shapes."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class StructuredInput(BaseModel):
    """Optional structured environmental variables a user can supply directly,
    either via the /chat/structured endpoint or embedded across multiple
    text turns. Every field is optional because the conversation layer is
    responsible for eliciting whatever is missing.
    """
    soil_organic_carbon: Optional[float] = Field(None, description="Percent, e.g. 0.3")
    soil_ph: Optional[float] = None
    soil_moisture: Optional[str] = Field(None, description="low | medium | high")
    rainfall: Optional[str] = Field(None, description="low | erratic | medium | high")
    temperature: Optional[str] = Field(None, description="stable | rising | extreme")
    land_use: Optional[str] = Field(None, description="monoculture | agroforestry | cleared | pasture | forest")
    species_richness: Optional[str] = Field(None, description="low | medium | high")
    habitat_diversity: Optional[str] = Field(None, description="low | medium | high")
    habitat_fragmentation: Optional[str] = Field(None, description="low | medium | high")
    pollution_level: Optional[str] = Field(None, description="low | medium | high")
    deforestation_rate: Optional[str] = Field(None, description="low | medium | high")
    region: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

    def known_fields(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ChatMessage(BaseModel):
    session_id: str
    message: str
    structured: Optional[StructuredInput] = None


class RecommendationOut(BaseModel):
    title: str
    what_to_do: str
    why_it_works: str
    impacted_metrics: List[str]
    expected_impact: str
    time_horizon: str
    confidence: str
    source: str
    linked_variables: List[str]
    retrieval_score: float


class ChatResponse(BaseModel):
    session_id: str
    reply_type: str  # "clarifying_question" | "recommendations" | "info"
    message: str
    missing_fields: List[str] = []
    known_fields: Dict[str, Any] = {}
    recommendations: List[RecommendationOut] = []
