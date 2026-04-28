import { ExternalLink, Globe2, Quote, ScrollText, ThumbsDown, ThumbsUp } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage, Citation } from '../types'

interface MessageBubbleProps {
  message: ChatMessage
  onInspectSources: (citations: Citation[]) => void
  onFeedback: (messageId: string, rating: 'up' | 'down') => void
}

function formatTime(value: string): string {
  try {
    return new Date(value).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  } catch {
    return value
  }
}

export default function MessageBubble({ message, onInspectSources, onFeedback }: MessageBubbleProps) {
  const isAssistant = message.role === 'assistant'
  const rating = message.feedback_rating || null
  const hasWebSources = message.citations.some((citation) => citation.source_type === 'web' && citation.url)
  const firstWebUrl = message.citations.find((citation) => citation.source_type === 'web' && citation.url)?.url || null

  return (
    <div className={`message-row ${message.role}`}>
      <div className={`bubble ${message.role}`}>
        <div className="bubble-header">
          <span className="bubble-role">{isAssistant ? 'ClarityAI' : 'You'}</span>
          <span className="bubble-time">{formatTime(message.created_at)}</span>
        </div>

        <div className="bubble-content markdown">
          {message.content ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          ) : (
            <div className="typing-placeholder">Working on your answer...</div>
          )}
        </div>

        {isAssistant && message.citations.length > 0 ? (
          <div className="citation-row">
            <button className="source-link-button" onClick={() => onInspectSources(message.citations)}>
              <Quote size={14} />
              View {message.citations.length} source{message.citations.length > 1 ? 's' : ''}
            </button>
            {message.citations.slice(0, 3).map((citation) => (
              <button key={citation.id} className="citation-chip" onClick={() => onInspectSources([citation])}>
                {citation.label}
              </button>
            ))}
            {hasWebSources ? (
              <span className="source-type-pill">
                <Globe2 size={12} /> web research
              </span>
            ) : (
              <span className="source-type-pill">
                <ScrollText size={12} /> uploaded knowledge
              </span>
            )}
            {firstWebUrl ? (
              <a className="source-link-button" href={firstWebUrl} target="_blank" rel="noreferrer">
                <ExternalLink size={14} />
                Open source
              </a>
            ) : null}
          </div>
        ) : null}

        {isAssistant ? (
          <div className="message-actions">
            <button
              className={`ghost-icon-button ${rating === 'up' ? 'selected' : ''}`}
              onClick={() => onFeedback(message.id, 'up')}
              aria-label="Mark as helpful"
            >
              <ThumbsUp size={14} />
            </button>
            <button
              className={`ghost-icon-button ${rating === 'down' ? 'selected' : ''}`}
              onClick={() => onFeedback(message.id, 'down')}
              aria-label="Mark as not helpful"
            >
              <ThumbsDown size={14} />
            </button>
          </div>
        ) : null}
      </div>
    </div>
  )
}
