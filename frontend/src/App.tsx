import { useEffect, useMemo, useRef, useState } from 'react'
import ChatView from './components/ChatView'
import Composer from './components/Composer'
import RightPanel from './components/RightPanel'
import Sidebar from './components/Sidebar'
import { api, streamChat } from './lib/api'
import type {
  ChatMessage,
  ChatMode,
  ChatSession,
  Citation,
  HealthInfo,
  ImageResult,
  ImagesInfo,
  KnowledgeDocument,
  ResearchInfo,
  ResearchMode,
  RouteInfo,
  SearchResponse
} from './types'

function makeTempMessage(role: 'user' | 'assistant', sessionId: string, content = ''): ChatMessage {
  return {
    id: crypto.randomUUID(),
    session_id: sessionId,
    role,
    content,
    citations: [],
    images: [],
    created_at: new Date().toISOString(),
    feedback_rating: null,
    feedback_note: null
  }
}

function sortSessions(items: ChatSession[]): ChatSession[] {
  return [...items].sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  return 'Something went wrong.'
}

function getInitialTheme(): 'dark' | 'light' {
  if (typeof window === 'undefined') return 'dark'

  try {
    const stored = window.localStorage.getItem('clarity-theme')
    return stored === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

function recordOf(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : null
}

function extractRouteInfo(value: unknown): RouteInfo | null {
  const record = recordOf(value)
  if (!record) return null

  const direct = record.route ?? record.route_info ?? null
  if (direct && typeof direct === 'object') return direct as RouteInfo

  const metadata = recordOf(record.metadata)
  const nested = metadata?.route ?? metadata?.route_info ?? null
  if (nested && typeof nested === 'object') return nested as RouteInfo

  return null
}

function extractResearchInfo(value: unknown): ResearchInfo | null {
  const record = recordOf(value)
  if (!record) return null

  const direct = record.research ?? record.research_info ?? null
  if (direct && typeof direct === 'object') return direct as ResearchInfo

  const metadata = recordOf(record.metadata)
  const nested = metadata?.research ?? metadata?.research_info ?? null
  if (nested && typeof nested === 'object') return nested as ResearchInfo

  return null
}

function extractSessionInsights(session: ChatSession): {
  route: RouteInfo | null
  research: ResearchInfo | null
} {
  const sessionRoute = extractRouteInfo(session)
  const sessionResearch = extractResearchInfo(session)

  if (sessionRoute || sessionResearch) {
    return {
      route: sessionRoute,
      research: sessionResearch
    }
  }

  const messages = Array.isArray(session.messages) ? session.messages : []
  const lastAssistant = [...messages].reverse().find((message) => message.role === 'assistant')

  return {
    route: extractRouteInfo(lastAssistant),
    research: extractResearchInfo(lastAssistant)
  }
}

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([])
  const [selectedCitations, setSelectedCitations] = useState<Citation[]>([])
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null)
  const [mode, setMode] = useState<ChatMode>('balanced')
  const [researchMode, setResearchMode] = useState<ResearchMode>('auto')
  const [busy, setBusy] = useState(false)
  const [uploadCount, setUploadCount] = useState(0)
  const [searching, setSearching] = useState(false)
  const [toast, setToast] = useState('')
  const [statusText, setStatusText] = useState('')
  const [routeInfo, setRouteInfo] = useState<RouteInfo | null>(null)
  const [researchInfo, setResearchInfo] = useState<ResearchInfo | null>(null)
  const [imagesInfo, setImagesInfo] = useState<ImagesInfo | null>(null)
  // Live images for the in-flight assistant message (before `done` arrives).
  const [liveImagesByMessageId, setLiveImagesByMessageId] = useState<Record<string, ImageResult[]>>({})
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [theme, setTheme] = useState<'dark' | 'light'>(getInitialTheme)

  const loadRequestRef = useRef(0)
  const searchRequestRef = useRef(0)
  const uploading = uploadCount > 0

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId),
    [sessions, activeSessionId]
  )

  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.dataset.theme = theme
    }

    if (typeof window !== 'undefined') {
      try {
        window.localStorage.setItem('clarity-theme', theme)
      } catch {
        // Ignore localStorage failures.
      }
    }
  }, [theme])

  useEffect(() => {
    void bootstrap()
  }, [])

  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(() => setToast(''), 3600)
    return () => window.clearTimeout(timer)
  }, [toast])

  function upsertSession(next: ChatSession) {
    setSessions((current) => {
      const filtered = current.filter((session) => session.id !== next.id)
      return sortSessions([{ ...next, messages: undefined }, ...filtered])
    })
  }

  async function bootstrap() {
    try {
      const healthInfo = await api.health().catch(() => null)
      setHealth(healthInfo)

      const [sessionList, documentList] = await Promise.all([
        api.listSessions().catch(() => []),
        api.listDocuments().catch(() => [])
      ])

      setSessions(sortSessions(sessionList))
      setDocuments(documentList)

      if (sessionList.length > 0) {
        await loadSession(sessionList[0].id)
      } else {
        const created = await api.createSession()
        setSessions([created])
        setActiveSessionId(created.id)
        setMessages([])
        setSelectedCitations([])
      }
    } catch (error) {
      setToast(errorMessage(error))
    }
  }

  async function loadSession(sessionId: string, options?: { preserveInsights?: boolean }) {
    const requestId = ++loadRequestRef.current
    const session = await api.getSession(sessionId)

    if (requestId !== loadRequestRef.current) return

    setActiveSessionId(session.id)
    setMessages(session.messages || [])
    setLiveImagesByMessageId({})

    const lastAssistant = [...(session.messages || [])].reverse().find((message) => message.role === 'assistant')
    setSelectedCitations(lastAssistant?.citations || [])
    setStatusText('')

    const insights = extractSessionInsights(session)
    if (!options?.preserveInsights || insights.route) {
      setRouteInfo(insights.route)
    }
    if (!options?.preserveInsights || insights.research) {
      setResearchInfo(insights.research)
    }
    if (!options?.preserveInsights) {
      setImagesInfo(null)
    }

    upsertSession(session)
  }

  async function handleNewSession() {
    try {
      const created = await api.createSession()
      setSessions((current) => sortSessions([created, ...current.filter((session) => session.id !== created.id)]))
      setActiveSessionId(created.id)
      setMessages([])
      setSelectedCitations([])
      setSearchResponse(null)
      setRouteInfo(null)
      setResearchInfo(null)
      setImagesInfo(null)
      setLiveImagesByMessageId({})
      setStatusText('')
    } catch (error) {
      setToast(errorMessage(error))
    }
  }

  async function handleDeleteSession(sessionId: string) {
    try {
      await api.deleteSession(sessionId)
      const remaining = sortSessions(sessions.filter((session) => session.id !== sessionId))
      setSessions(remaining)

      if (activeSessionId !== sessionId) return

      if (remaining[0]) {
        await loadSession(remaining[0].id)
        return
      }

      await handleNewSession()
    } catch (error) {
      setToast(errorMessage(error))
    }
  }

  async function handleSend(text: string) {
    if (busy) return

    const trimmed = text.trim()
    if (!trimmed) return

    let finalSessionId = activeSessionId
    let tempAssistantId = ''

    setBusy(true)
    setSelectedCitations([])
    setRouteInfo(null)
    setResearchInfo(null)
    setImagesInfo(null)
    setStatusText('Preparing response')

    try {
      let sessionId = activeSessionId
      if (!sessionId) {
        const created = await api.createSession()
        sessionId = created.id
        finalSessionId = created.id
        setSessions((current) => sortSessions([created, ...current.filter((session) => session.id !== created.id)]))
        setActiveSessionId(created.id)
      }

      const tempUser = makeTempMessage('user', sessionId, trimmed)
      const tempAssistant = makeTempMessage('assistant', sessionId, '')
      tempAssistantId = tempAssistant.id

      setMessages((current) => [...current, tempUser, tempAssistant])

      await streamChat(
        {
          session_id: sessionId,
          message: trimmed,
          mode,
          research_mode: researchMode
        },
        {
          onMeta: async (payload) => {
            finalSessionId = payload.session.id
            upsertSession(payload.session)
            setSelectedCitations(payload.citations || [])
            setRouteInfo(payload.route)
            setResearchInfo(payload.research)
            setImagesInfo(payload.images || null)
            // If meta already includes images (rare but possible), seed them.
            if (payload.images?.results?.length) {
              setLiveImagesByMessageId((current) => ({
                ...current,
                [tempAssistant.id]: payload.images.results
              }))
            }
            setMessages((current) =>
              current.map((message) =>
                message.id === tempAssistant.id
                  ? {
                      ...message,
                      citations: payload.citations || [],
                      images: payload.images?.results || []
                    }
                  : message
              )
            )
          },
          onStatus: async (payload) => {
            setStatusText(payload.message)
          },
          onImages: async (payload) => {
            const incoming = payload.results || []
            setLiveImagesByMessageId((current) => ({
              ...current,
              [tempAssistant.id]: incoming
            }))
            setImagesInfo((current) => ({
              attempted: true,
              count: incoming.length,
              error: current?.error ?? null,
              results: incoming
            }))
            setMessages((current) =>
              current.map((message) =>
                message.id === tempAssistant.id ? { ...message, images: incoming } : message
              )
            )
          },
          onToken: async (token) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === tempAssistant.id ? { ...message, content: `${message.content}${token}` } : message
              )
            )
          },
          onDone: async (payload) => {
            finalSessionId = payload.session.id
            upsertSession(payload.session)
            setSelectedCitations(payload.citations || payload.message.citations || [])
            setRouteInfo(payload.route)
            setResearchInfo(payload.research)
            setImagesInfo(payload.images || null)
            setMessages((current) =>
              current.map((message) =>
                message.id === tempAssistant.id
                  ? {
                      ...payload.message,
                      images: payload.message.images || payload.images?.results || []
                    }
                  : message
              )
            )
            // The persisted message now carries images on its own; we can
            // drop the live cache for this id.
            setLiveImagesByMessageId((current) => {
              if (!(tempAssistant.id in current)) return current
              const next = { ...current }
              delete next[tempAssistant.id]
              return next
            })
            setStatusText('')
          }
        }
      )

      await loadSession(finalSessionId, { preserveInsights: true })
    } catch (error) {
      if (tempAssistantId) {
        setMessages((current) =>
          current.map((message) =>
            message.id === tempAssistantId ? { ...message, content: `Error: ${errorMessage(error)}` } : message
          )
        )
      }
      setToast(errorMessage(error))
      setStatusText('')
      setRouteInfo(null)
      setResearchInfo(null)
      setImagesInfo(null)
    } finally {
      setBusy(false)
    }
  }

  async function handleUpload(file: File) {
    setUploadCount((current) => current + 1)

    try {
      const document = await api.uploadDocument(file)
      setDocuments((current) => [document, ...current.filter((item) => item.id !== document.id)])
      setToast(`Uploaded ${document.source_name}`)
    } catch (error) {
      setToast(errorMessage(error))
    } finally {
      setUploadCount((current) => Math.max(0, current - 1))
    }
  }

  async function handleSearchKnowledge(query: string) {
    const trimmed = query.trim()
    if (!trimmed) {
      searchRequestRef.current += 1
      setSearchResponse(null)
      setSearching(false)
      return
    }

    const requestId = ++searchRequestRef.current

    try {
      setSearching(true)
      const result = await api.searchKnowledge(trimmed)
      if (requestId !== searchRequestRef.current) return
      setSearchResponse(result)
    } catch (error) {
      if (requestId === searchRequestRef.current) {
        setToast(errorMessage(error))
      }
    } finally {
      if (requestId === searchRequestRef.current) {
        setSearching(false)
      }
    }
  }

  async function handleDeleteDocument(documentId: string) {
    try {
      await api.deleteDocument(documentId)
      setDocuments((current) => current.filter((document) => document.id !== documentId))
      setSearchResponse(null)
      setToast('Document deleted.')
    } catch (error) {
      setToast(errorMessage(error))
    }
  }

  async function handleReindex() {
    try {
      const result = await api.reindexKnowledge()
      const dense = result.dense ? ` Dense retrieval: ${result.dense}.` : ''
      setToast(`Knowledge index rebuilt.${dense}`)
    } catch (error) {
      setToast(errorMessage(error))
    }
  }

  async function handleFeedback(messageId: string, rating: 'up' | 'down') {
    try {
      const updated = (await api.sendFeedback(messageId, rating)) as ChatMessage
      setMessages((current) => current.map((message) => (message.id === updated.id ? { ...updated, images: message.images } : message)))
    } catch (error) {
      setToast(errorMessage(error))
    }
  }

  return (
    <div className="app-shell">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        theme={theme}
        health={health}
        onSelect={(sessionId) => {
          if (busy) return
          void loadSession(sessionId).catch((error) => setToast(errorMessage(error)))
        }}
        onNewSession={() => void handleNewSession()}
        onDeleteSession={(sessionId) => void handleDeleteSession(sessionId)}
        onToggleTheme={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}
      />

      <main className="center-column">
        <ChatView
          title={activeSession?.title || 'New conversation'}
          messages={messages}
          liveImagesByMessageId={liveImagesByMessageId}
          statusText={statusText}
          routeInfo={routeInfo}
          researchInfo={researchInfo}
          imagesInfo={imagesInfo}
          onInspectSources={setSelectedCitations}
          onFeedback={(messageId, rating) => void handleFeedback(messageId, rating)}
          onSendSuggestion={(text) => void handleSend(text)}
        />
        <Composer
          mode={mode}
          researchMode={researchMode}
          busy={busy}
          setMode={setMode}
          setResearchMode={setResearchMode}
          onSend={(text) => void handleSend(text)}
        />
      </main>

      <RightPanel
        citations={selectedCitations}
        routeInfo={routeInfo}
        researchInfo={researchInfo}
        documents={documents}
        searchResponse={searchResponse}
        uploading={uploading}
        searching={searching}
        onUpload={(file) => void handleUpload(file)}
        onSearch={(query) => void handleSearchKnowledge(query)}
        onDeleteDocument={(documentId) => void handleDeleteDocument(documentId)}
        onReindex={() => void handleReindex()}
      />

      {toast ? <div className="toast">{toast}</div> : null}
    </div>
  )
}
