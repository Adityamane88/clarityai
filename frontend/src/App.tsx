import { useEffect, useMemo, useState } from 'react'
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
  const [uploading, setUploading] = useState(false)
  const [searching, setSearching] = useState(false)
  const [toast, setToast] = useState('')
  const [statusText, setStatusText] = useState('')
  const [routeInfo, setRouteInfo] = useState<RouteInfo | null>(null)
  const [researchInfo, setResearchInfo] = useState<ResearchInfo | null>(null)
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const stored = localStorage.getItem('clarity-theme')
    return stored === 'light' ? 'light' : 'dark'
  })

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId),
    [sessions, activeSessionId]
  )

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('clarity-theme', theme)
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
      const [sessionList, documentList, healthInfo] = await Promise.all([
        api.listSessions(),
        api.listDocuments(),
        api.health().catch(() => null)
      ])
      setSessions(sortSessions(sessionList))
      setDocuments(documentList)
      setHealth(healthInfo)
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

  async function loadSession(sessionId: string) {
    const session = await api.getSession(sessionId)
    setActiveSessionId(session.id)
    setMessages(session.messages || [])
    const lastAssistant = [...(session.messages || [])].reverse().find((message) => message.role === 'assistant')
    setSelectedCitations(lastAssistant?.citations || [])
    setStatusText('')
    upsertSession(session)
  }

  async function handleNewSession() {
    try {
      const created = await api.createSession()
      setSessions((current) => sortSessions([created, ...current]))
      setActiveSessionId(created.id)
      setMessages([])
      setSelectedCitations([])
      setSearchResponse(null)
      setRouteInfo(null)
      setResearchInfo(null)
      setStatusText('')
    } catch (error) {
      setToast(errorMessage(error))
    }
  }

  async function handleDeleteSession(sessionId: string) {
    const confirmed = window.confirm('Delete this session?')
    if (!confirmed) return
    try {
      await api.deleteSession(sessionId)
      const remaining = sessions.filter((session) => session.id !== sessionId)
      setSessions(remaining)
      if (activeSessionId === sessionId) {
        if (remaining[0]) {
          await loadSession(remaining[0].id)
        } else {
          await handleNewSession()
        }
      }
    } catch (error) {
      setToast(errorMessage(error))
    }
  }

  async function handleSend(text: string) {
    if (busy) return

    let sessionId = activeSessionId
    if (!sessionId) {
      const created = await api.createSession()
      sessionId = created.id
      setSessions((current) => sortSessions([created, ...current]))
      setActiveSessionId(created.id)
    }

    const tempUser = makeTempMessage('user', sessionId, text)
    const tempAssistant = makeTempMessage('assistant', sessionId, '')
    setMessages((current) => [...current, tempUser, tempAssistant])
    setSelectedCitations([])
    setStatusText('Preparing response')
    setBusy(true)

    let finalSessionId = sessionId

    try {
      await streamChat(
        {
          session_id: sessionId,
          message: text,
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
            setMessages((current) =>
              current.map((message) =>
                message.id === tempAssistant.id ? { ...message, citations: payload.citations || [] } : message
              )
            )
          },
          onStatus: async (payload) => {
            setStatusText(payload.message)
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
            setMessages((current) => current.map((message) => (message.id === tempAssistant.id ? payload.message : message)))
            setStatusText('')
          }
        }
      )
      await loadSession(finalSessionId)
    } catch (error) {
      setMessages((current) =>
        current.map((message) =>
          message.id === tempAssistant.id ? { ...message, content: `Error: ${errorMessage(error)}` } : message
        )
      )
      setToast(errorMessage(error))
      setStatusText('')
    } finally {
      setBusy(false)
    }
  }

  async function handleUpload(file: File) {
    try {
      setUploading(true)
      const document = await api.uploadDocument(file)
      setDocuments((current) => [document, ...current.filter((item) => item.id !== document.id)])
      setToast(`Uploaded ${document.source_name}`)
    } catch (error) {
      setToast(errorMessage(error))
    } finally {
      setUploading(false)
    }
  }

  async function handleSearchKnowledge(query: string) {
    if (!query.trim()) {
      setSearchResponse(null)
      return
    }
    try {
      setSearching(true)
      const result = await api.searchKnowledge(query)
      setSearchResponse(result)
    } catch (error) {
      setToast(errorMessage(error))
    } finally {
      setSearching(false)
    }
  }

  async function handleDeleteDocument(documentId: string) {
    const confirmed = window.confirm('Delete this document?')
    if (!confirmed) return
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
      setMessages((current) => current.map((message) => (message.id === updated.id ? updated : message)))
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
        onSelect={(sessionId) => void loadSession(sessionId)}
        onNewSession={() => void handleNewSession()}
        onDeleteSession={(sessionId) => void handleDeleteSession(sessionId)}
        onToggleTheme={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}
      />

      <main className="center-column">
        <ChatView
          title={activeSession?.title || 'New conversation'}
          messages={messages}
          statusText={statusText}
          routeInfo={routeInfo}
          researchInfo={researchInfo}
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
