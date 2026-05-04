import {
  ArrowDown,
  BookOpen,
  Compass,
  ImageIcon,
  MessagesSquare,
  SearchCheck,
  ShieldCheck,
  Sparkles
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ChatMessage, Citation, ImageResult, ImagesInfo, ResearchInfo, RouteInfo } from '../types'
import MessageBubble from './MessageBubble'

interface ChatViewProps {
  title: string
  messages: ChatMessage[]
  liveImagesByMessageId?: Record<string, ImageResult[]>
  statusText: string
  routeInfo: RouteInfo | null
  researchInfo: ResearchInfo | null
  imagesInfo?: ImagesInfo | null
  onInspectSources: (citations: Citation[]) => void
  onFeedback: (messageId: string, rating: 'up' | 'down') => void
  onSendSuggestion: (text: string) => void
}

const suggestions = [
  'Summarize the key principles in my uploaded documents and cite the strongest evidence.',
  'Compare two ways to handle customer escalations, explain the tradeoffs, and recommend one.',
  'Show me images of golden retrievers and tell me a bit about the breed.',
  'Help me turn a messy situation into a clear plan for this week with realistic next steps.'
]

const AUTO_SCROLL_THRESHOLD = 120

function humanize(value: string | null | undefined): string {
  if (!value) return ''
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (match) => match.toUpperCase())
}

function routeLabel(routeInfo: RouteInfo | null): string {
  const route = routeInfo?.resolved_route || routeInfo?.route
  if (route === 'hybrid') return 'Hybrid answer'
  if (route === 'research') return 'Researched answer'
  if (route === 'local') return 'Knowledge-grounded answer'
  if (route === 'direct') return 'Direct answer'
  if (route === 'safe_completion') return 'Safety-aware answer'
  return 'Ready'
}

function isNearBottom(node: HTMLDivElement): boolean {
  const distance = node.scrollHeight - node.scrollTop - node.clientHeight
  return distance <= AUTO_SCROLL_THRESHOLD
}

function usePrefersReducedMotion(): boolean {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return

    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setPrefersReducedMotion(mediaQuery.matches)
    update()

    if (typeof mediaQuery.addEventListener === 'function') {
      mediaQuery.addEventListener('change', update)
      return () => mediaQuery.removeEventListener('change', update)
    }

    mediaQuery.addListener(update)
    return () => mediaQuery.removeListener(update)
  }, [])

  return prefersReducedMotion
}

function countUniqueSources(messages: ChatMessage[]): number {
  const seen = new Set<string>()

  for (const message of messages) {
    if (message.role !== 'assistant' || !Array.isArray(message.citations)) continue

    for (const citation of message.citations) {
      const typedCitation = citation as Citation & {
        number?: number | string | null
        chunk_id?: string
        document_id?: string
        url?: string
        label?: string
      }
      const key =
        typedCitation.id ||
        typedCitation.chunk_id ||
        typedCitation.document_id ||
        typedCitation.url ||
        (typedCitation.number !== null && typedCitation.number !== undefined ? String(typedCitation.number) : '') ||
        typedCitation.label

      if (key) seen.add(String(key))
    }
  }

  return seen.size
}

export default function ChatView({
  title,
  messages,
  liveImagesByMessageId,
  statusText,
  routeInfo,
  researchInfo,
  imagesInfo,
  onInspectSources,
  onFeedback,
  onSendSuggestion
}: ChatViewProps) {
  const safeMessages = useMemo(() => (Array.isArray(messages) ? messages.filter(Boolean) : []), [messages])
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)
  const previousRef = useRef({
    messageCount: 0,
    lastMessageId: '',
    lastMessageContent: '',
    statusText: ''
  })
  const [stickToBottom, setStickToBottom] = useState(true)
  const [showJumpToLatest, setShowJumpToLatest] = useState(false)
  const prefersReducedMotion = usePrefersReducedMotion()

  const latestAssistantMessage = useMemo(() => {
    return [...safeMessages].reverse().find((message) => message.role === 'assistant') || null
  }, [safeMessages])

  const latestCitationCount = Array.isArray(latestAssistantMessage?.citations)
    ? latestAssistantMessage.citations.length
    : 0
  const researchedSourceCount =
    typeof researchInfo?.count === 'number'
      ? researchInfo.count
      : latestAssistantMessage?.citations?.filter((citation) => citation.source_type === 'web').length || 0
  const imageCount =
    typeof imagesInfo?.count === 'number'
      ? imagesInfo.count
      : latestAssistantMessage?.images?.length || 0
  const uniqueSourceCount = useMemo(() => countUniqueSources(safeMessages), [safeMessages])
  const routeReason = humanize(routeInfo?.reason || null)

  const scrollToLatest = useCallback(
    (behavior?: ScrollBehavior) => {
      endRef.current?.scrollIntoView({
        behavior: behavior ?? (prefersReducedMotion ? 'auto' : 'smooth'),
        block: 'end'
      })
    },
    [prefersReducedMotion]
  )

  const handleScroll = useCallback(() => {
    const node = scrollRef.current
    if (!node) return

    const nearBottom = isNearBottom(node)
    setStickToBottom(nearBottom)
    setShowJumpToLatest(!nearBottom && safeMessages.length > 0)
  }, [safeMessages.length])

  useEffect(() => {
    const lastMessage = safeMessages[safeMessages.length - 1]
    const previous = previousRef.current
    const messageAdded = previous.messageCount !== safeMessages.length || previous.lastMessageId !== (lastMessage?.id || '')
    const contentChanged = previous.lastMessageContent !== (lastMessage?.content || '')
    const statusChanged = previous.statusText !== statusText

    previousRef.current = {
      messageCount: safeMessages.length,
      lastMessageId: lastMessage?.id || '',
      lastMessageContent: lastMessage?.content || '',
      statusText
    }

    if (stickToBottom) {
      const behavior: ScrollBehavior = messageAdded && !prefersReducedMotion ? 'smooth' : 'auto'
      if ((messageAdded || contentChanged || statusChanged) && typeof window !== 'undefined') {
        const frame = window.requestAnimationFrame(() => scrollToLatest(behavior))
        return () => window.cancelAnimationFrame(frame)
      }
      setShowJumpToLatest(false)
    } else if (messageAdded || statusChanged) {
      setShowJumpToLatest(true)
    }
  }, [prefersReducedMotion, safeMessages, scrollToLatest, statusText, stickToBottom])

  useEffect(() => {
    handleScroll()
  }, [handleScroll])

  return (
    <section className="chat-view panel" aria-label="Conversation view">
      <div className="chat-header">
        <div>
          <div className="chat-title">{title || 'New conversation'}</div>
          <div className="chat-subtitle">
            Crisp answers first, evidence when it matters, images when they help, and citations you can actually inspect.
          </div>
        </div>
        <div className="status-pill-group" aria-label="Workspace capabilities">
          <span className="status-pill"><ShieldCheck size={14} /> Safety layer</span>
          <span className="status-pill"><BookOpen size={14} /> Source citations</span>
          <span className="status-pill"><ImageIcon size={14} /> Image search</span>
          <span className="status-pill"><Sparkles size={14} /> Multi-turn memory</span>
        </div>
      </div>

      <div className="hero-strip" aria-live="polite">
        <div className="hero-chip hero-chip-primary">
          <Compass size={15} />
          <span>{routeLabel(routeInfo)}</span>
        </div>
        <div className="hero-chip">
          <SearchCheck size={15} />
          <span>
            {researchInfo?.attempted
              ? `${researchedSourceCount} researched source${researchedSourceCount === 1 ? '' : 's'}`
              : 'No web research used yet'}
          </span>
        </div>
        <div className="hero-chip">
          <MessagesSquare size={15} />
          <span>{safeMessages.length} message{safeMessages.length === 1 ? '' : 's'}</span>
        </div>
        {imageCount > 0 ? (
          <div className="hero-chip">
            <ImageIcon size={15} />
            <span>{imageCount} image{imageCount === 1 ? '' : 's'} attached</span>
          </div>
        ) : null}
        {latestCitationCount > 0 ? (
          <div className="hero-chip">
            <BookOpen size={15} />
            <span>{latestCitationCount} cited snippet{latestCitationCount === 1 ? '' : 's'} in the latest answer</span>
          </div>
        ) : null}
        {uniqueSourceCount > 0 ? (
          <div className="hero-chip">
            <BookOpen size={15} />
            <span>{uniqueSourceCount} unique source{uniqueSourceCount === 1 ? '' : 's'} used</span>
          </div>
        ) : null}
        {routeReason ? <div className="hero-progress">Reason: {routeReason}</div> : null}
        {statusText ? (
          <div className="hero-progress" role="status" aria-live="polite">
            {statusText}
          </div>
        ) : null}
      </div>

      <div
        ref={scrollRef}
        className="messages-scroll"
        onScroll={handleScroll}
        role="log"
        aria-live="polite"
        aria-relevant="additions text"
        aria-busy={Boolean(statusText)}
      >
        {safeMessages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-hero">
              Ask anything. The assistant can combine your uploaded knowledge, conversation context, optional web research, and images into one grounded answer.
            </div>
            <div className="empty-copy">
              It is strongest at synthesis, troubleshooting, comparisons, planning, and source-grounded responses. Open the right panel to upload files or inspect evidence behind any answer.
            </div>
            <div className="suggestion-grid">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  className="suggestion-card"
                  type="button"
                  onClick={() => onSendSuggestion(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {safeMessages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            liveImages={liveImagesByMessageId?.[message.id]}
            onInspectSources={onInspectSources}
            onFeedback={onFeedback}
          />
        ))}
        <div ref={endRef} aria-hidden="true" />
      </div>

      {showJumpToLatest ? (
        <button
          className="jump-to-latest-button"
          type="button"
          onClick={() => {
            setStickToBottom(true)
            setShowJumpToLatest(false)
            scrollToLatest()
          }}
          aria-label="Jump to the latest message"
        >
          <ArrowDown size={14} />
          Jump to latest
        </button>
      ) : null}
    </section>
  )
}
