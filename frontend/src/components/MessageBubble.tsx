import { Check, Copy, ExternalLink, Globe2, Quote, ScrollText, ThumbsDown, ThumbsUp } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage, Citation } from '../types'

interface MessageBubbleProps {
  message: ChatMessage
  onInspectSources: (citations: Citation[]) => void
  onFeedback: (messageId: string, rating: 'up' | 'down') => void
}

type CitationLike = Citation & {
  number?: number | string | null
  citation_number?: number | string | null
}

function formatTime(value: string | null | undefined): string {
  if (!value) return ''
  try {
    return new Date(value).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  } catch {
    return value
  }
}

function isSafeHref(href: string | null | undefined): href is string {
  if (!href) return false
  if (href.startsWith('#') || href.startsWith('/')) return true

  try {
    const url = new URL(href, 'https://example.com')
    return ['http:', 'https:', 'mailto:', 'tel:'].includes(url.protocol)
  } catch {
    return false
  }
}

async function copyText(value: string): Promise<boolean> {
  if (!value) return false

  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value)
      return true
    }
  } catch {
    // Fall through to the legacy copy path.
  }

  if (typeof document === 'undefined') return false

  try {
    const textarea = document.createElement('textarea')
    textarea.value = value
    textarea.setAttribute('readonly', 'true')
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    textarea.style.pointerEvents = 'none'
    document.body.appendChild(textarea)
    textarea.focus()
    textarea.select()
    const copied = document.execCommand('copy')
    document.body.removeChild(textarea)
    return copied
  } catch {
    return false
  }
}

function asCitations(value: ChatMessage['citations'] | null | undefined): CitationLike[] {
  return Array.isArray(value) ? (value.filter(Boolean) as CitationLike[]) : []
}

function citationKey(citation: CitationLike, index: number): string {
  return String(
    citation.id ||
      citation.chunk_id ||
      citation.document_id ||
      citation.url ||
      citation.number ||
      citation.citation_number ||
      citation.label ||
      index
  )
}

function citationNumber(citation: CitationLike): string | null {
  const raw = citation.number ?? citation.citation_number ?? null
  if (raw === null || raw === undefined) return null
  const text = String(raw).trim()
  return text ? text : null
}

function dedupeCitations(citations: CitationLike[]): CitationLike[] {
  const seen = new Set<string>()
  const deduped: CitationLike[] = []

  citations.forEach((citation, index) => {
    const key = citationKey(citation, index)
    if (seen.has(key)) return
    seen.add(key)
    deduped.push(citation)
  })

  return deduped.sort((left, right) => {
    const leftNumber = Number(citationNumber(left) || '')
    const rightNumber = Number(citationNumber(right) || '')
    const leftHasNumber = Number.isFinite(leftNumber)
    const rightHasNumber = Number.isFinite(rightNumber)

    if (leftHasNumber && rightHasNumber && leftNumber !== rightNumber) {
      return leftNumber - rightNumber
    }
    if (leftHasNumber) return -1
    if (rightHasNumber) return 1

    return (right.score ?? 0) - (left.score ?? 0)
  })
}

function citationBadge(citation: CitationLike): string {
  const number = citationNumber(citation)
  if (number) return `[${number}]`
  if (citation.label?.trim()) return citation.label.trim()
  return citation.source_type === 'web' ? 'Web' : 'Source'
}

function citationTitle(citation: CitationLike): string {
  return citation.document_title || citation.source_name || 'Untitled source'
}

function sourceMeta(citation: CitationLike): string {
  return [citation.source_name, citation.page_label ? `page ${citation.page_label}` : '', citation.published_at]
    .filter(Boolean)
    .join(' | ')
}

function CodeBlock({ className, children }: { className?: string; children: unknown }) {
  const [copied, setCopied] = useState(false)
  const code = String(children ?? '').replace(/\n$/, '')

  useEffect(() => {
    if (!copied) return undefined
    if (typeof window === 'undefined') return undefined
    const timer = window.setTimeout(() => setCopied(false), 1400)
    return () => window.clearTimeout(timer)
  }, [copied])

  return (
    <div className="code-block-wrap">
      <div className="code-block-toolbar">
        <span className="code-block-language">{className?.replace('language-', '') || 'code'}</span>
        <button
          className="ghost-icon-button"
          type="button"
          onClick={async () => {
            const ok = await copyText(code)
            if (ok) setCopied(true)
          }}
          aria-label="Copy code block"
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
        </button>
      </div>
      <pre className="markdown-code-block">
        <code className={className}>{code}</code>
      </pre>
    </div>
  )
}

const markdownComponents = {
  a(props: any) {
    const { href, children, ...rest } = props
    if (!isSafeHref(href)) return <span>{children}</span>

    const external = !href.startsWith('#') && !href.startsWith('/')

    return (
      <a {...rest} href={href} target={external ? '_blank' : undefined} rel={external ? 'noreferrer noopener' : undefined}>
        {children}
      </a>
    )
  },
  code(props: any) {
    const { inline, className, children, ...rest } = props

    if (inline) {
      return (
        <code className={className} {...rest}>
          {children}
        </code>
      )
    }

    return <CodeBlock className={className}>{children}</CodeBlock>
  },
  table(props: any) {
    return (
      <div className="markdown-table-wrap">
        <table {...props} />
      </div>
    )
  }
}

export default function MessageBubble({ message, onInspectSources, onFeedback }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false)
  const [expandedSources, setExpandedSources] = useState(false)

  const isAssistant = message.role === 'assistant'
  const rating = message.feedback_rating || null
  const citations = useMemo(() => dedupeCitations(asCitations(message.citations)), [message.citations])
  const webCitations = useMemo(
    () => citations.filter((citation) => citation.source_type === 'web' && isSafeHref(citation.url)),
    [citations]
  )
  const knowledgeCount = citations.length - webCitations.length
  const firstWebUrl = webCitations[0]?.url ?? null
  const visibleCitations = expandedSources ? citations : citations.slice(0, 4)
  const content = typeof message.content === 'string' ? message.content : ''
  const hasContent = Boolean(content.trim())

  useEffect(() => {
    if (!copied) return undefined
    if (typeof window === 'undefined') return undefined

    const timeoutId = window.setTimeout(() => setCopied(false), 1600)
    return () => window.clearTimeout(timeoutId)
  }, [copied])

  async function handleCopy() {
    const ok = await copyText(content)
    if (ok) setCopied(true)
  }

  return (
    <article className={`message-row ${message.role}`} aria-busy={isAssistant && !hasContent}>
      <div className={`bubble ${message.role}`}>
        <div className="bubble-header">
          <span className="bubble-role">{isAssistant ? 'ClarityAI' : 'You'}</span>
          <time className="bubble-time" dateTime={message.created_at}>
            {formatTime(message.created_at)}
          </time>
        </div>

        <div className="bubble-content markdown">
          {hasContent ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {content}
            </ReactMarkdown>
          ) : (
            <div className="typing-placeholder">Working on your answer...</div>
          )}
        </div>

        {isAssistant && citations.length > 0 ? (
          <div className="citation-row">
            <button className="source-link-button" type="button" onClick={() => onInspectSources(citations)}>
              <Quote size={14} />
              View {citations.length} source{citations.length === 1 ? '' : 's'}
            </button>

            {visibleCitations.map((citation, index) => (
              <button
                key={citationKey(citation, index)}
                className="citation-chip"
                type="button"
                title={`${citationTitle(citation)}${sourceMeta(citation) ? ` - ${sourceMeta(citation)}` : ''}`}
                onClick={() => onInspectSources([citation])}
              >
                {citationBadge(citation)}
              </button>
            ))}

            {citations.length > visibleCitations.length ? (
              <button className="citation-chip" type="button" onClick={() => setExpandedSources(true)}>
                +{citations.length - visibleCitations.length} more
              </button>
            ) : null}

            {expandedSources && citations.length > 4 ? (
              <button className="citation-chip" type="button" onClick={() => setExpandedSources(false)}>
                Show less
              </button>
            ) : null}

            {webCitations.length > 0 ? (
              <span className="source-type-pill" title="Web sources used in this answer">
                <Globe2 size={12} /> {webCitations.length} web
              </span>
            ) : null}

            {knowledgeCount > 0 ? (
              <span className="source-type-pill" title="Uploaded knowledge sources used in this answer">
                <ScrollText size={12} /> {knowledgeCount} knowledge
              </span>
            ) : null}

            {firstWebUrl ? (
              <a className="source-link-button" href={firstWebUrl} target="_blank" rel="noreferrer noopener">
                <ExternalLink size={14} />
                Open top source
              </a>
            ) : null}
          </div>
        ) : null}

        {isAssistant ? (
          <div className="message-actions">
            <button
              className={`ghost-icon-button ${copied ? 'selected' : ''}`}
              onClick={handleCopy}
              aria-label={copied ? 'Copied answer' : 'Copy answer'}
              type="button"
              disabled={!hasContent}
              title={copied ? 'Copied' : 'Copy answer'}
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </button>
            <button
              className={`ghost-icon-button ${rating === 'up' ? 'selected' : ''}`}
              onClick={() => onFeedback(message.id, 'up')}
              aria-label="Mark as helpful"
              aria-pressed={rating === 'up'}
              type="button"
              title="Helpful"
            >
              <ThumbsUp size={14} />
            </button>
            <button
              className={`ghost-icon-button ${rating === 'down' ? 'selected' : ''}`}
              onClick={() => onFeedback(message.id, 'down')}
              aria-label="Mark as not helpful"
              aria-pressed={rating === 'down'}
              type="button"
              title="Not helpful"
            >
              <ThumbsDown size={14} />
            </button>
          </div>
        ) : null}
      </div>
    </article>
  )
}
