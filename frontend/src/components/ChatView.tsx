import { BookOpen, Compass, SearchCheck, ShieldCheck, Sparkles } from 'lucide-react'
import { useEffect, useRef } from 'react'
import type { ChatMessage, Citation, ResearchInfo, RouteInfo } from '../types'
import MessageBubble from './MessageBubble'

interface ChatViewProps {
  title: string
  messages: ChatMessage[]
  statusText: string
  routeInfo: RouteInfo | null
  researchInfo: ResearchInfo | null
  onInspectSources: (citations: Citation[]) => void
  onFeedback: (messageId: string, rating: 'up' | 'down') => void
  onSendSuggestion: (text: string) => void
}

const suggestions = [
  'What are the key principles in the sample documents I have loaded? Cite the snippets you use.',
  'Compare two approaches I might use to handle customer escalations, with tradeoffs and a recommendation.',
  'Walk me through how to debug a slow API endpoint, step by step. Be specific about what to check first.',
  'I am feeling overwhelmed at work. Help me think through what to drop, defer, and do this week.'
]

function routeLabel(routeInfo: RouteInfo | null): string {
  if (!routeInfo) return 'Ready'
  const route = routeInfo.resolved_route || routeInfo.route
  if (route === 'hybrid') return 'Hybrid answer'
  if (route === 'research') return 'Researched answer'
  if (route === 'local') return 'Knowledge-grounded answer'
  return 'Ready'
}

export default function ChatView({
  title,
  messages,
  statusText,
  routeInfo,
  researchInfo,
  onInspectSources,
  onFeedback,
  onSendSuggestion
}: ChatViewProps) {
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, statusText])

  return (
    <section className="chat-view panel">
      <div className="chat-header">
        <div>
          <div className="chat-title">{title || 'New conversation'}</div>
          <div className="chat-subtitle">Thoughtful, source-grounded answers with problem-solving structure and empathy.</div>
        </div>
        <div className="status-pill-group">
          <span className="status-pill"><ShieldCheck size={14} /> Safety layer</span>
          <span className="status-pill"><BookOpen size={14} /> Source citations</span>
          <span className="status-pill"><Sparkles size={14} /> Multi-turn memory</span>
        </div>
      </div>

      <div className="hero-strip">
        <div className="hero-chip hero-chip-primary">
          <Compass size={15} />
          <span>{routeLabel(routeInfo)}</span>
        </div>
        <div className="hero-chip">
          <SearchCheck size={15} />
          <span>
            {researchInfo?.attempted ? `${researchInfo.count} researched source${researchInfo.count === 1 ? '' : 's'}` : 'No web research used yet'}
          </span>
        </div>
        {statusText ? <div className="hero-progress">{statusText}</div> : null}
      </div>

      <div className="messages-scroll">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-hero">Ask anything. ClarityAI grounds answers in your uploaded documents and cites every claim.</div>
            <div className="empty-copy">
              Two sample documents are already loaded so you can try it immediately. Upload your own files in the right panel to make it knowledgeable about your domain. For live web facts, enable web research in <code>backend/.env</code>.
            </div>
            <div className="suggestion-grid">
              {suggestions.map((suggestion) => (
                <button key={suggestion} className="suggestion-card" onClick={() => onSendSuggestion(suggestion)}>
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            onInspectSources={onInspectSources}
            onFeedback={onFeedback}
          />
        ))}
        <div ref={endRef} />
      </div>
    </section>
  )
}
