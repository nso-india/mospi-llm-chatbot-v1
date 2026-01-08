from beanie import Document, init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import Field
from typing import Optional, List
from datetime import datetime
from typing import Literal
# ==== MongoDB Models ====

class Interaction(Document):
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    query: str
    response: str
    sources: Optional[List[str]] = []
    feedback: Optional[Literal["like", "dislike"]] = None  # ✅ NEW FIELD

    class Settings:
        name = "interactions"

class SessionAnalytics(Document):
    session_id: str
    chat_count: int
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    avg_query_length: float
    avg_response_length: float
    fallback_count: int

    class Settings:
        name = "session_analytics"


# ==== Initialization Function ====

async def init_db():
    client = AsyncIOMotorClient("mongodb://admin:RhrPq%213%40qr%2AAfz%247%237@localhost:27017/admin")
    await init_beanie(
        database=client.mospi_db,
        document_models=[Interaction, SessionAnalytics]
    )
