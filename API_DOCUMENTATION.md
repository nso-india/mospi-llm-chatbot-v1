# MoSPI Chatbot API Documentation

## Base URL
```
http://localhost:8095
```

## Authentication
All endpoints require API key in header:
```
access_token: YOUR_API_KEY
```

## Endpoints

### Session Management

#### Start Session
```
GET /start_session
```
**Response:**
```json
{
  "session_id": "uuid-string"
}
```

#### Delete Session
```
DELETE /delete_session/{session_id}
```

### Chat

#### Ask Question
```
POST /ask
```
**Body:**
```json
{
  "query": "Your question here",
  "session_id": "session-uuid"
}
```

### File Management

#### Upload File
```
POST /upload_file
```
**Body:** Form-data with file (PDF or JSON)

#### List Documents
```
GET /list_documents
```

#### Get Document Chunks
```
GET /get_chunks_by_docname?doc_name=document_name
```

#### Delete Document Chunks
```
DELETE /delete_chunks_by_docname?doc_name=document_name
```

### URL Management

#### List URLs
```
GET /list_url_titles
```

#### Get URL Chunks
```
GET /get_chunks_by_url?url=url_here
```

### Feedback

#### Set Feedback
```
POST /interactions/{interaction_id}/feedback
```
**Body:**
```json
{
  "feedback": "like" // or "dislike"
}
```

#### Get Feedback Stats
```
GET /feedback_stats
```

### Analytics

#### Get All Interactions
```
GET /interactions
```

#### Get Session Stats
```
GET /session_stats
```

### Logs

#### Get Chatbot Logs
```
GET /chatbot_logs?lines=200
```

#### Download Logs
```
GET /download_chatbot_logs
```