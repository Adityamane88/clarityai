export type ChatMode = 'balanced' | 'concise' | 'deep'
export type ResearchMode = 'auto' | 'off' | 'force'
export type MessageRole = 'user' | 'assistant'

export interface Citation {
  id: string
  label: string
  chunk_id?: number | null
  document_id: string
  document_title: string
  source_name: string
  page_label?: string | null
  snippet: string
  content: string
  score: number
  source_type?: 'knowledge' | 'web'
  url?: string | null
  published_at?: string | null
}

export interface ChatMessage {
  id: string
  session_id: string
  role: MessageRole
  content: string
  citations: Citation[]
  feedback_rating?: 'up' | 'down' | null
  feedback_note?: string | null
  created_at: string
}

export interface ChatSession {
  id: string
  title: string
  summary: string
  created_at: string
  updated_at: string
  messages?: ChatMessage[]
}

export interface KnowledgeDocument {
  id: string
  title: string
  source_name: string
  mime_type: string
  text_preview: string
  chunk_count: number
  created_at: string
  updated_at: string
}

export interface SearchResponse {
  query: string
  count: number
  confidence: number
  results: Citation[]
}

export interface RouteInfo {
  route: string
  resolved_route: string
  needs_web_research: boolean
  needs_local_knowledge: boolean
  reason: string
  query_is_time_sensitive: boolean
}

export interface ResearchInfo {
  attempted: boolean
  count: number
  error?: string | null
}

export interface StreamMeta {
  session: ChatSession
  citations: Citation[]
  search: {
    confidence: number
    hits: number
    dense_used?: boolean
  }
  safety: {
    severity: string
    blocked: boolean
    message?: string | null
    reason?: string | null
  }
  route: RouteInfo
  research: ResearchInfo
}

export interface StreamDone {
  message: ChatMessage
  citations: Citation[]
  session: ChatSession
  safety: StreamMeta['safety']
  route: RouteInfo
  research: ResearchInfo
}

export interface StreamStatus {
  stage: 'retrieving' | 'researching' | 'answering'
  message: string
}

export interface HealthInfo {
  status: string
  remote_llm_configured: boolean
  chat_model: string | null
  dense_retrieval_enabled: boolean
  web_research_configured: boolean
}
