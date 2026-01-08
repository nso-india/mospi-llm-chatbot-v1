from fastapi import FastAPI, HTTPException, Depends, Security, File, UploadFile, APIRouter, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from datetime import datetime
import uuid
import shutil
import os
import tempfile
from starlette.status import HTTP_403_FORBIDDEN
from dotenv import load_dotenv
import os
from fastapi.responses import PlainTextResponse
from fastapi import APIRouter, HTTPException
import re
from fastapi import Body
from datetime import datetime
from chatbot import info_vectordb
from typing import List, Optional

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "access_token"

# API key security (header only)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return API_KEY
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Could not validate API key",
        )






# Import your existing modules
from init_db import initialize_database
from models import Interaction
from chatbot import (
    initialize_components,
    handle_question,
    get_chunks_for_doc,
    memory_sessions,
    create_memory,
    list_all_urls,
    get_chunks_for_url,
    cleanup_expired_sessions,
    list_all_documents,
    delete_chunks_for_doc,
    update_doc_metadata
)
from create_chunks import (
    add_all_pdfs_with_tables_and_text_chunks,
    process_json_files,
)

# Define models
class QueryRequest(BaseModel):
    query: str
    session_id: str

class SessionResponse(BaseModel):
    session_id: str

# Create FastAPI app
app = FastAPI(title="MOSPI AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import os
from fastapi.staticfiles import StaticFiles
from config import STATIC_DIRECTORY, STATIC_ROUTE, STATIC_NAME

# Ensure static folder exists
if not os.path.exists(STATIC_DIRECTORY):
    os.makedirs(STATIC_DIRECTORY)

# Mount static files
app.mount(STATIC_ROUTE, StaticFiles(directory=STATIC_DIRECTORY), name=STATIC_NAME)

# Router with global API key dependency (header only)
router = APIRouter(dependencies=[Depends(get_api_key)])

@app.on_event("startup")
async def startup_event():
    initialize_components()
    await initialize_database()

@router.get("/start_session", response_model=SessionResponse)
async def create_session():
    cleanup_expired_sessions()
    session_id = str(uuid.uuid4())
    memory_sessions[session_id] = (datetime.now(), create_memory())
    return {"session_id": session_id}

@router.post("/ask")
async def ask_question(request: QueryRequest):
    cleanup_expired_sessions()
    return await handle_question(request)





from pydantic import BaseModel
from models import Interaction
from typing import Literal

class FeedbackRequest(BaseModel):
    feedback: Literal["like", "dislike"]

@router.post("/interactions/{interaction_id}/feedback")
async def set_interaction_feedback(interaction_id: str, request: FeedbackRequest):
    interaction = await Interaction.get(interaction_id)
    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")

    interaction.feedback = request.feedback
    await interaction.save()
    return {
        "message": "Feedback recorded successfully",
        "interaction_id": interaction_id,
        "feedback": request.feedback
    }

@router.get("/feedback_stats")
async def get_feedback_stats():
    likes = await Interaction.find(Interaction.feedback == "like").count()
    dislikes = await Interaction.find(Interaction.feedback == "dislike").count()
    total = await Interaction.find_all().count()
    return {
        "total_interactions": total,
        "likes": likes,
        "dislikes": dislikes,
        "like_percentage": round((likes / total) * 100, 2) if total > 0 else 0
    }


@router.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.lower().endswith(".pdf"):
            add_all_pdfs_with_tables_and_text_chunks(os.path.dirname(file_path))
        elif file.filename.lower().endswith(".json"):
            process_json_files(os.path.dirname(file_path))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        return {"message": f"File {file.filename} processed successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        shutil.rmtree(temp_dir)

@router.get("/list_documents")
async def list_documents():
    return list_all_documents()

@router.get("/get_chunks_by_docname")
async def get_chunks_by_docname(doc_name: str):
    return get_chunks_for_doc(doc_name)

@router.delete("/delete_chunks_by_docname")
async def delete_chunks_by_docname(doc_name: str):
    doc_names = [d.strip() for d in doc_name.split(",")]
    return delete_chunks_for_doc(doc_names)

@router.get("/list_url_titles")
async def list_all_urls_api():
    return list_all_urls()

@router.get("/get_chunks_by_url")
async def get_chunks_by_url(url: str):
    return get_chunks_for_url(url)


@router.get("/interactions")
async def get_all_interactions():
    logs = await Interaction.find_all().sort("-timestamp").to_list()
    return [
        {
            "session_id": i.session_id,
            "timestamp": i.timestamp,
            "query": i.query,
            "response": i.response,
            "sources": i.sources or []
        }
        for i in logs
    ]

@router.get("/session_stats")
async def get_session_statistics():
    fallback_messages = [
        "This seems to be outside my scope. Unfortunately, I am unable to assist you with your requested query. Thank you for your understanding.",
        "यह मेरे दायरे से बाहर लगता है। दुर्भाग्य से, मैं आपके अनुरोधित प्रश्न में सहायता नहीं कर सकता। धन्यवाद।"
    ]
    interactions = await Interaction.find_all().to_list()
    sessions = {}

    for i in interactions:
        sid = i.session_id
        sessions.setdefault(sid, {
            "session_id": sid,
            "chat_count": 0,
            "queries": [],
            "responses": [],
            "start_time": i.timestamp,
            "end_time": i.timestamp,
            "fallback_count": 0
        })

        session = sessions[sid]
        session["chat_count"] += 1
        session["queries"].append(i.query)
        session["responses"].append(i.response)
        session["start_time"] = min(session["start_time"], i.timestamp)
        session["end_time"] = max(session["end_time"], i.timestamp)
        if i.response.strip() in fallback_messages:
            session["fallback_count"] += 1

    results = []
    for session in sessions.values():
        duration = (session["end_time"] - session["start_time"]).total_seconds() / 60.0
        avg_query_length = sum(len(q) for q in session["queries"]) / len(session["queries"])
        avg_response_length = sum(len(r) for r in session["responses"]) / len(session["responses"])
        results.append({
            "session_id": session["session_id"],
            "chat_count": session["chat_count"],
            "start_time": session["start_time"],
            "end_time": session["end_time"],
            "duration_minutes": round(duration, 2),
            "avg_query_length": round(avg_query_length, 2),
            "avg_response_length": round(avg_response_length, 2),
            "fallback_count": session["fallback_count"]
        })

    return results

@router.delete("/delete_session/{session_id}")
async def delete_session(session_id: str):
    if session_id in memory_sessions:
        del memory_sessions[session_id]
    deleted = await Interaction.find(Interaction.session_id == session_id).delete()
    return {
        "message": f"Session {session_id} deleted successfully.",
        "interactions_deleted": deleted
    }




CHATBOT_LOG_FILE = "logs/chatbot.log"

@router.get("/chatbot_logs")
async def get_chatbot_logs(lines: int = 200):
    """
    Return the last N lines of chatbot logs.
    Default = 200 lines.
    """
    if not os.path.exists(CHATBOT_LOG_FILE):
        raise HTTPException(status_code=404, detail="Chatbot log file not found")

    try:
        with open(CHATBOT_LOG_FILE, "r") as f:
            content = f.readlines()

        # tail N lines
        last_lines = content[-lines:] if lines > 0 else content
        return PlainTextResponse("".join(last_lines))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading chatbot log file: {str(e)}")


@router.get("/download_chatbot_logs")
async def download_chatbot_logs():
    """
    Download the full chatbot log file.
    """
    if not os.path.exists(CHATBOT_LOG_FILE):
        raise HTTPException(status_code=404, detail="Chatbot log file not found")

    return PlainTextResponse(
        open(CHATBOT_LOG_FILE, "r").read(),
        headers={
            "Content-Disposition": "attachment; filename=chatbot.log"
        },
        media_type="text/plain"
    )



class MetadataUpdateByDocRequest(BaseModel):
    doc_names: List[str]
    uploaded_at: Optional[str] = None


@router.post("/update_metadata_by_doc")
async def update_metadata_by_doc(request: MetadataUpdateByDocRequest):
    """
    API endpoint → calls chatbot.py function to update metadata
    """
    try:
        return update_doc_metadata(
            doc_names=request.doc_names,
            uploaded_at=request.uploaded_at
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update metadata: {str(e)}")

from chatbot import info_vectordb, whoswho_names, load_whoswho_names
@app.on_event("startup")
async def startup_event():
    initialize_components()
    await initialize_database()

    global whoswho_names
    retriever = info_vectordb
    if retriever is not None:
        whoswho_chunks_all = retriever.as_retriever(search_type="all").invoke(
            "", filter={"doc_name": "Who 's who.json"}
        )
        whoswho_names.update(load_whoswho_names(whoswho_chunks_all))
        print(f"✅ Preloaded {len(whoswho_names)} officer names")

# Include router so all endpoints require API key in header
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
