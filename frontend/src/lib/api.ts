import type {
  ChatMode,
  ChatSession,
  KnowledgeDocument,
  ResearchMode,
  SearchResponse,
  StreamDone,
  StreamMeta,
  StreamStatus
} from '../types'

// Default to relative URLs so Vite's dev-server proxy (and nginx in prod) forwards
// /api and /health to the backend. This sidesteps CORS entirely, regardless of which
// port Vite ends up on. Set VITE_API_BASE_URL only if you serve the frontend from a
// different origin than the backend.
const API_BASE = ((import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')) + '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {})
    }
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Request failed with ${response.status}`)
  }

  return response.json() as Promise<T>
}

export const api = {
  health: () =>
    fetch(`${API_BASE.replace(/\/api$/, '')}/health`).then(async (response) => {
      if (!response.ok) throw new Error(`Health check failed with ${response.status}`)
      return response.json() as Promise<{
        status: string
        remote_llm_configured: boolean
        chat_model: string | null
        dense_retrieval_enabled: boolean
        web_research_configured: boolean
      }>
    }),
  listSessions: () => request<ChatSession[]>('/sessions'),
  createSession: (title?: string) =>
    request<ChatSession>('/sessions', {
      method: 'POST',
      body: JSON.stringify(title ? { title } : {})
    }),
  getSession: (sessionId: string) => request<ChatSession>(`/sessions/${sessionId}`),
  deleteSession: (sessionId: string) =>
    request<{ status: string; session_id: string }>(`/sessions/${sessionId}`, { method: 'DELETE' }),
  listDocuments: () => request<KnowledgeDocument[]>('/knowledge/documents'),
  searchKnowledge: (query: string) => request<SearchResponse>(`/knowledge/search?q=${encodeURIComponent(query)}`),
  reindexKnowledge: () => request<{ status: string; chunks: number; dense?: string }>('/knowledge/reindex', { method: 'POST' }),
  deleteDocument: (documentId: string) =>
    request<{ status: string; document_id: string }>(`/knowledge/documents/${documentId}`, { method: 'DELETE' }),
  sendFeedback: (messageId: string, rating: 'up' | 'down') =>
    request(`/feedback/messages/${messageId}`, {
      method: 'POST',
      body: JSON.stringify({ rating })
    }),
  uploadDocument: async (file: File): Promise<KnowledgeDocument> => {
    const formData = new FormData()
    formData.append('file', file)
    const response = await fetch(`${API_BASE}/knowledge/upload`, {
      method: 'POST',
      body: formData
    })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || `Upload failed with ${response.status}`)
    }
    return response.json() as Promise<KnowledgeDocument>
  }
}

type StreamHandlers = {
  onMeta?: (payload: StreamMeta) => void | Promise<void>
  onStatus?: (payload: StreamStatus) => void | Promise<void>
  onToken?: (token: string) => void | Promise<void>
  onDone?: (payload: StreamDone) => void | Promise<void>
  onEvent?: (eventName: string, payload: unknown) => void | Promise<void>
}

export async function streamChat(
  payload: { session_id?: string | null; message: string; mode: ChatMode; research_mode: ResearchMode },
  handlers: StreamHandlers
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  })

  if (!response.ok || !response.body) {
    const text = await response.text()
    throw new Error(text || `Stream failed with ${response.status}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const segments = buffer.split('\n\n')
    buffer = segments.pop() || ''

    for (const segment of segments) {
      if (!segment.trim()) continue
      const lines = segment.split('\n')
      let eventName = 'message'
      let data = ''

      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventName = line.slice(6).trim()
        }
        if (line.startsWith('data:')) {
          data += line.slice(5).trim()
        }
      }

      if (!data) continue
      const parsed = JSON.parse(data)
      await handlers.onEvent?.(eventName, parsed)

      if (eventName === 'meta') await handlers.onMeta?.(parsed as StreamMeta)
      if (eventName === 'status') await handlers.onStatus?.(parsed as StreamStatus)
      if (eventName === 'token') await handlers.onToken?.((parsed as { content?: string }).content || '')
      if (eventName === 'done') await handlers.onDone?.(parsed as StreamDone)
    }
  }
}
