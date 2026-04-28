import {
  AlertCircle,
  Bot,
  BrainCircuit,
  CheckCircle2,
  MessagesSquare,
  MoonStar,
  Plus,
  ShieldCheck,
  SunMedium,
  Trash2
} from 'lucide-react'
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
  const llmConnected = health?.remote_llm_configured ?? false
  const researchOn = health?.web_research_configured ?? false

  return (
    <aside className="sidebar panel">
      <div className="brand-block">
        <div className="brand-row">
          <div className="brand-icon"><Bot size={18} /></div>
          <div>
            <div className="brand-title">ClarityAI</div>
            <div className="brand-subtitle">Research-grade conversation workspace</div>
          </div>
        </div>

        <div className="brand-signal-grid">
          <div className={`brand-signal-card ${llmConnected ? 'ok' : 'warn'}`}>
            {llmConnected ? <CheckCircle2 size={15} /> : <AlertCircle size={15} />}
            <span>{llmConnected ? `LLM: ${health?.chat_model || 'connected'}` : 'LLM not connected'}</span>
          </div>
          <div className={`brand-signal-card ${researchOn ? 'ok' : ''}`}>
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
            Set <code>LLM_API_KEY</code> in <code>backend/.env</code> for real model answers.
            Restart the backend after editing.
          </div>
        ) : null}
      </div>

      <div className="sidebar-actions">
        <button className="primary-button" onClick={onNewSession}>
          <Plus size={16} />
          New chat
        </button>
        <button className="icon-button" onClick={onToggleTheme} aria-label="Toggle theme">
          {theme === 'dark' ? <SunMedium size={16} /> : <MoonStar size={16} />}
        </button>
      </div>

      <div className="sidebar-section-label">Sessions</div>
      <div className="session-list">
        {sessions.length === 0 ? (
          <div className="empty-mini-card">
            <MessagesSquare size={18} />
            <span>No sessions yet</span>
          </div>
        ) : null}

        {sessions.map((session) => (
          <div
            key={session.id}
            className={`session-item ${session.id === activeSessionId ? 'active' : ''}`}
            onClick={() => onSelect(session.id)}
          >
            <div className="session-copy">
              <div className="session-title">{session.title}</div>
              <div className="session-meta">{formatTime(session.updated_at)}</div>
            </div>
            <button
              className="ghost-icon-button"
              onClick={(event) => {
                event.stopPropagation()
                onDeleteSession(session.id)
              }}
              aria-label="Delete session"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>

      <div className="sidebar-footnote">
        Use <strong>Auto research</strong> for smart routing, <strong>Knowledge only</strong> for internal answers, and{' '}
        <strong>Force research</strong> when you want external evidence.
      </div>
    </aside>
  )
}
