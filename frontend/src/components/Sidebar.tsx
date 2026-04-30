import {
  AlertCircle,
  Bot,
  BrainCircuit,
  CheckCircle2,
  MessagesSquare,
  MoonStar,
  Plus,
  Search,
  ShieldCheck,
  SunMedium,
  Trash2
} from 'lucide-react'
import { useMemo, useState, type ChangeEvent, type KeyboardEvent } from 'react'
import type { ChatSession, HealthInfo } from '../types'

interface SidebarProps {
  sessions: ChatSession[]
  activeSessionId: string
  theme: 'dark' | 'light'
  health: HealthInfo | null
  onSelect: (sessionId: string) => void
  onNewSession: () => void
  onDeleteSession: (sessionId: string) => void
  onToggleTheme: () => void
}

type SessionLike = ChatSession & {
  preview?: string
  last_message_preview?: string
  message_count?: number
}

function formatTime(value: string): string {
  try {
    return new Date(value).toLocaleString([], {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit'
    })
  } catch {
    return value
  }
}

function formatRelativeTime(value: string): string {
  try {
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return value

    const diffMs = date.getTime() - Date.now()
    const abs = Math.abs(diffMs)
    const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })

    if (abs < 60 * 60 * 1000) {
      return formatter.format(Math.round(diffMs / (60 * 1000)), 'minute')
    }
    if (abs < 24 * 60 * 60 * 1000) {
      return formatter.format(Math.round(diffMs / (60 * 60 * 1000)), 'hour')
    }
    if (abs < 30 * 24 * 60 * 60 * 1000) {
      return formatter.format(Math.round(diffMs / (24 * 60 * 60 * 1000)), 'day')
    }
    return formatTime(value)
  } catch {
    return value
  }
}

function getHealthField(health: HealthInfo | null, field: string): unknown {
  if (!health || typeof health !== 'object') return undefined
  return (health as unknown as Record<string, unknown>)[field]
}

function getSessionPreview(session: SessionLike): string {
  const preview = session.preview || session.last_message_preview
  if (preview && preview.trim()) return preview
  return 'Open this conversation'
}

function safeTitle(session: SessionLike): string {
  return session.title?.trim() || 'Untitled conversation'
}

function sortSessions(sessions: ChatSession[]): SessionLike[] {
  return [...sessions].sort((left, right) => {
    const leftTime = new Date(left.updated_at).getTime()
    const rightTime = new Date(right.updated_at).getTime()
    if (Number.isNaN(leftTime) || Number.isNaN(rightTime)) return 0
    return rightTime - leftTime
  }) as SessionLike[]
}

export default function Sidebar({
  sessions,
  activeSessionId,
  theme,
  health,
  onSelect,
  onNewSession,
  onDeleteSession,
  onToggleTheme
}: SidebarProps) {
  const [query, setQuery] = useState('')

  const llmConnected = Boolean(health?.remote_llm_configured)
  const researchOn = Boolean(health?.web_research_configured)
  const chatModel = typeof getHealthField(health, 'chat_model') === 'string' ? String(getHealthField(health, 'chat_model')) : null
  const embeddingModel =
    typeof getHealthField(health, 'embedding_model') === 'string'
      ? String(getHealthField(health, 'embedding_model'))
      : null
  const healthError =
    typeof getHealthField(health, 'error') === 'string'
      ? String(getHealthField(health, 'error'))
      : typeof getHealthField(health, 'startup_error') === 'string'
        ? String(getHealthField(health, 'startup_error'))
        : null

  const filteredSessions = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const sorted = sortSessions(sessions)
    if (!needle) return sorted

    return sorted.filter((session) => {
      const haystack = `${safeTitle(session)} ${getSessionPreview(session)} ${session.id}`.toLowerCase()
      return haystack.includes(needle)
    })
  }, [query, sessions])

  function handleListKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (!['ArrowDown', 'ArrowUp'].includes(event.key) || filteredSessions.length === 0) return
    event.preventDefault()
    const currentIndex = filteredSessions.findIndex((session) => session.id === activeSessionId)
    const fallbackIndex = currentIndex === -1 ? 0 : currentIndex
    const nextIndex =
      event.key === 'ArrowDown'
        ? (fallbackIndex + 1) % filteredSessions.length
        : (fallbackIndex - 1 + filteredSessions.length) % filteredSessions.length
    onSelect(filteredSessions[nextIndex].id)
  }

  return (
    <aside className="sidebar panel" aria-label="Conversations and workspace status">
      <div className="brand-block">
        <div className="brand-row">
          <div className="brand-icon">
            <Bot size={18} />
          </div>
          <div>
            <div className="brand-title">ClarityAI</div>
            <div className="brand-subtitle">Research-grade conversation workspace</div>
          </div>
        </div>

        <div className="brand-signal-grid">
          <div className={`brand-signal-card ${llmConnected ? 'ok' : 'warn'}`} title={chatModel || 'Model status'}>
            {llmConnected ? <CheckCircle2 size={15} /> : <AlertCircle size={15} />}
            <span>{llmConnected ? `LLM: ${chatModel || 'connected'}` : 'LLM not connected'}</span>
          </div>
          <div
            className={`brand-signal-card ${researchOn ? 'ok' : ''}`}
            title={embeddingModel ? `Embeddings: ${embeddingModel}` : 'Research and retrieval status'}
          >
            <BrainCircuit size={15} />
            <span>{researchOn ? 'Web research on' : 'Hybrid knowledge'}</span>
          </div>
          <div className="brand-signal-card">
            <ShieldCheck size={15} />
            <span>Safe by design</span>
          </div>
        </div>

        {!llmConnected ? (
          <div className="brand-warning">
            Set <code>LLM_API_KEY</code> in <code>backend/.env</code> for live model answers, then restart the backend.
          </div>
        ) : null}

        {healthError ? <div className="brand-warning">{healthError}</div> : null}
      </div>

      <div className="sidebar-actions">
        <button className="primary-button" onClick={onNewSession} type="button">
          <Plus size={16} />
          New chat
        </button>
        <button className="icon-button" onClick={onToggleTheme} aria-label="Toggle theme" type="button">
          {theme === 'dark' ? <SunMedium size={16} /> : <MoonStar size={16} />}
        </button>
      </div>

      <div className="panel-section-row">
        <div className="sidebar-section-label">Sessions</div>
        <div className="panel-helper-text">
          {filteredSessions.length}/{sessions.length}
        </div>
      </div>

      <div className="search-field-wrap sidebar-search">
        <Search size={14} />
        <input
          className="search-field"
          value={query}
          onChange={(event: ChangeEvent<HTMLInputElement>) => setQuery(event.target.value)}
          placeholder="Search chats"
          aria-label="Search conversations"
        />
      </div>

      <div className="session-list" onKeyDown={handleListKeyDown} role="list" aria-label="Conversation list">
        {filteredSessions.length === 0 ? (
          <div className="empty-mini-card">
            <MessagesSquare size={18} />
            <span>{query ? 'No matching sessions' : 'No sessions yet'}</span>
          </div>
        ) : null}

        {filteredSessions.map((session) => {
          const active = session.id === activeSessionId
          const title = safeTitle(session)
          const exactTime = formatTime(session.updated_at)
          const relativeTime = formatRelativeTime(session.updated_at)

          return (
            <div key={session.id} className={`session-item ${active ? 'active' : ''}`}>
              <button
                className="session-select-button"
                onClick={() => onSelect(session.id)}
                type="button"
                aria-current={active ? 'page' : undefined}
                title={title}
              >
                <div className="session-copy">
                  <div className="session-title">{title}</div>
                  <div className="session-meta" title={exactTime}>{relativeTime}</div>
                  <div className="session-preview">{getSessionPreview(session)}</div>
                </div>
              </button>
              <button
                className="ghost-icon-button"
                onClick={() => {
                  const ok = typeof window === 'undefined' || window.confirm(`Delete \"${title}\"?`)
                  if (ok) onDeleteSession(session.id)
                }}
                aria-label={`Delete ${title}`}
                type="button"
                title="Delete chat"
              >
                <Trash2 size={14} />
              </button>
            </div>
          )
        })}
      </div>

      <div className="sidebar-footnote">
        Use <strong>Auto research</strong> for smart routing, <strong>Knowledge only</strong> for internal answers, and{' '}
        <strong>Force research</strong> when you need external evidence.
      </div>
    </aside>
  )
}
