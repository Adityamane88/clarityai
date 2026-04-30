import {
  BookOpen,
  Check,
  Copy,
  Database,
  ExternalLink,
  FileUp,
  Globe2,
  Quote,
  RotateCcw,
  Search,
  Trash2,
  X
} from 'lucide-react'
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type KeyboardEvent
} from 'react'
import type { Citation, KnowledgeDocument, ResearchInfo, RouteInfo, SearchResponse } from '../types'

interface RightPanelProps {
  citations: Citation[]
  routeInfo: RouteInfo | null
  researchInfo: ResearchInfo | null
  documents: KnowledgeDocument[]
  searchResponse: SearchResponse | null
  uploading: boolean
  searching: boolean
  onUpload: (file: File) => void
  onSearch: (query: string) => void
  onDeleteDocument: (documentId: string) => void
  onReindex: () => void
}

type RichCitation = Citation & {
  number?: number | string | null
  citation_number?: number | string | null
}

type PanelTab = 'sources' | 'knowledge'
type SourceFilter = 'all' | 'web' | 'knowledge'

const TAB_STORAGE_KEY = 'clarityai:right-panel:tab:elite'
const QUERY_STORAGE_KEY = 'clarityai:right-panel:query:elite'
const DOC_FILTER_STORAGE_KEY = 'clarityai:right-panel:doc-filter:elite'
const ACCEPTED_FILE_TYPES = '.pdf,.txt,.md,.markdown,.csv,.json,.docx,.doc,.html,.htm'

function getStoredValue(key: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback
  try {
    return window.localStorage.getItem(key) || fallback
  } catch {
    return fallback
  }
}

function getInitialTab(): PanelTab {
  return getStoredValue(TAB_STORAGE_KEY, 'sources') === 'knowledge' ? 'knowledge' : 'sources'
}

function getInitialQuery(): string {
  return getStoredValue(QUERY_STORAGE_KEY, '')
}

function getInitialDocumentFilter(): string {
  return getStoredValue(DOC_FILTER_STORAGE_KEY, '')
}

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
  if (route === 'hybrid') return 'Hybrid'
  if (route === 'research') return 'Research'
  if (route === 'local') return 'Knowledge'
  return humanize(route || 'ready') || 'Ready'
}

function citationNumber(citation: RichCitation): number | null {
  const raw = citation.number ?? citation.citation_number ?? null
  if (typeof raw === 'number' && Number.isFinite(raw)) return raw
  if (typeof raw === 'string') {
    const parsed = Number(raw)
    if (Number.isFinite(parsed)) return parsed
  }
  const match = citation.label?.match(/\[?(\d+)\]?/)
  if (!match) return null
  const parsed = Number(match[1])
  return Number.isFinite(parsed) ? parsed : null
}

function sourceKey(citation: RichCitation, index: number): string {
  return String(
    citation.id ||
      citation.chunk_id ||
      citation.document_id ||
      citation.url ||
      citation.number ||
      citation.citation_number ||
      `${citation.label || citation.source_name || 'source'}-${index}`
  )
}

function sourceLabel(citation: RichCitation, index: number): string {
  const number = citationNumber(citation)
  if (number !== null) return `[${number}]`

  const label = citation.label?.trim()
  if (label && !/^\[?[sw]\d+\]?$/i.test(label)) return label
  return `Source ${index + 1}`
}

function sourceTitle(citation: RichCitation): string {
  return citation.document_title || citation.source_name || citation.label || 'Untitled source'
}

function formatMaybeDate(value: string | null | undefined): string {
  if (!value) return ''
  try {
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return value
    return date.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' })
  } catch {
    return value
  }
}

function sourceMeta(citation: RichCitation): string {
  return [citation.source_name, citation.page_label ? `page ${citation.page_label}` : '', formatMaybeDate(citation.published_at)]
    .filter(Boolean)
    .join(' | ')
}

function formatScore(score: number | null | undefined): string | null {
  return typeof score === 'number' && Number.isFinite(score) ? score.toFixed(2) : null
}

function normalizeText(value: string | null | undefined): string {
  return (value || '').replace(/\s+/g, ' ').trim()
}

function dedupeAndSortCitations(citations: Citation[]): RichCitation[] {
  const items = Array.isArray(citations) ? (citations as RichCitation[]) : []
  const seen = new Set<string>()
  const deduped: RichCitation[] = []

  items.forEach((citation, index) => {
    const key = sourceKey(citation, index)
    if (seen.has(key)) return
    seen.add(key)
    deduped.push(citation)
  })

  return deduped.sort((left, right) => {
    const leftNumber = citationNumber(left)
    const rightNumber = citationNumber(right)

    if (leftNumber !== null && rightNumber !== null && leftNumber !== rightNumber) {
      return leftNumber - rightNumber
    }
    if (leftNumber !== null) return -1
    if (rightNumber !== null) return 1

    const leftScore = typeof left.score === 'number' ? left.score : -1
    const rightScore = typeof right.score === 'number' ? right.score : -1
    if (leftScore !== rightScore) {
      return rightScore - leftScore
    }

    return sourceTitle(left).localeCompare(sourceTitle(right))
  })
}

function documentLabel(document: KnowledgeDocument): string {
  return document.title || document.source_name || 'Untitled document'
}

async function copyText(value: string): Promise<boolean> {
  if (!value) return false

  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value)
      return true
    }
  } catch {
    // Fall through to legacy copy.
  }

  if (typeof document === 'undefined') return false
  try {
    const textarea = document.createElement('textarea')
    textarea.value = value
    textarea.setAttribute('readonly', 'true')
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.select()
    const copied = document.execCommand('copy')
    document.body.removeChild(textarea)
    return copied
  } catch {
    return false
  }
}

export default function RightPanel({
  citations,
  routeInfo,
  researchInfo,
  documents,
  searchResponse,
  uploading,
  searching,
  onUpload,
  onSearch,
  onDeleteDocument,
  onReindex
}: RightPanelProps) {
  const [tab, setTab] = useState<PanelTab>(getInitialTab)
  const [query, setQuery] = useState(getInitialQuery)
  const [documentFilter, setDocumentFilter] = useState(getInitialDocumentFilter)
  const [dragActive, setDragActive] = useState(false)
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('all')
  const [copied, setCopied] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const normalizedQuery = query.trim()
  const normalizedDocumentFilter = documentFilter.trim().toLowerCase()
  const canSearch = normalizedQuery.length > 0 && !searching
  const sortedCitations = useMemo(() => dedupeAndSortCitations(citations), [citations])
  const visibleCitations = useMemo(() => {
    if (sourceFilter === 'web') return sortedCitations.filter((citation) => citation.source_type === 'web')
    if (sourceFilter === 'knowledge') return sortedCitations.filter((citation) => citation.source_type !== 'web')
    return sortedCitations
  }, [sortedCitations, sourceFilter])
  const sortedDocuments = useMemo(() => {
    const base = [...documents].sort((left, right) => documentLabel(left).localeCompare(documentLabel(right)))
    if (!normalizedDocumentFilter) return base

    return base.filter((document) => {
      const haystack = `${documentLabel(document)} ${document.source_name || ''} ${document.text_preview || ''}`.toLowerCase()
      return haystack.includes(normalizedDocumentFilter)
    })
  }, [documents, normalizedDocumentFilter])

  const webCount = sortedCitations.filter((citation) => citation.source_type === 'web').length
  const knowledgeCount = sortedCitations.length - webCount
  const totalChunks = documents.reduce((sum, document) => sum + (document.chunk_count || 0), 0)

  useEffect(() => {
    if (!copied || typeof window === 'undefined') return undefined
    const timer = window.setTimeout(() => setCopied(false), 1500)
    return () => window.clearTimeout(timer)
  }, [copied])

  function persistTab(nextTab: PanelTab) {
    setTab(nextTab)
    if (typeof window === 'undefined') return
    try {
      window.localStorage.setItem(TAB_STORAGE_KEY, nextTab)
    } catch {
      // Ignore storage failures.
    }
  }

  function updateQuery(nextValue: string) {
    setQuery(nextValue)
    if (typeof window === 'undefined') return
    try {
      if (nextValue) {
        window.localStorage.setItem(QUERY_STORAGE_KEY, nextValue)
      } else {
        window.localStorage.removeItem(QUERY_STORAGE_KEY)
      }
    } catch {
      // Ignore storage failures.
    }
  }

  function updateDocumentFilter(nextValue: string) {
    setDocumentFilter(nextValue)
    if (typeof window === 'undefined') return
    try {
      if (nextValue) {
        window.localStorage.setItem(DOC_FILTER_STORAGE_KEY, nextValue)
      } else {
        window.localStorage.removeItem(DOC_FILTER_STORAGE_KEY)
      }
    } catch {
      // Ignore storage failures.
    }
  }

  function uploadFiles(files: FileList | File[] | null | undefined) {
    const fileList = Array.from(files || []).filter((file) => file.size > 0)
    if (fileList.length === 0) return

    persistTab('knowledge')
    fileList.forEach((file, index) => {
      if (typeof window === 'undefined') {
        onUpload(file)
        return
      }
      window.setTimeout(() => onUpload(file), index * 100)
    })
  }

  function handleDrag(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    event.stopPropagation()
  }

  async function handleCopySources() {
    const payload = visibleCitations
      .map((citation, index) => {
        const lines = [sourceLabel(citation, index), sourceTitle(citation)]
        const meta = sourceMeta(citation)
        const snippet = normalizeText(citation.snippet)
        if (meta) lines.push(meta)
        if (snippet) lines.push(snippet)
        return lines.join('\n')
      })
      .join('\n\n')

    const ok = await copyText(payload)
    if (ok) setCopied(true)
  }

  function runSearch() {
    if (!canSearch) return
    persistTab('knowledge')
    onSearch(normalizedQuery)
  }

  function handleKnowledgeKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Enter') {
      event.preventDefault()
      runSearch()
    }
    if (event.key === 'Escape' && query) {
      updateQuery('')
    }
  }

  return (
    <aside className="right-panel panel" aria-label="Sources and knowledge panel">
      <div className="tab-row" role="tablist" aria-label="Right panel tabs">
        <button
          className={`tab-button ${tab === 'sources' ? 'active' : ''}`}
          type="button"
          role="tab"
          aria-selected={tab === 'sources'}
          onClick={() => persistTab('sources')}
        >
          <Quote size={14} />
          Sources {sortedCitations.length > 0 ? `(${sortedCitations.length})` : ''}
        </button>
        <button
          className={`tab-button ${tab === 'knowledge' ? 'active' : ''}`}
          type="button"
          role="tab"
          aria-selected={tab === 'knowledge'}
          onClick={() => persistTab('knowledge')}
        >
          <Database size={14} />
          Knowledge {documents.length > 0 ? `(${documents.length})` : ''}
        </button>
      </div>

      {tab === 'sources' ? (
        <div className="panel-scroll" role="tabpanel" aria-label="Sources">
          <div className="source-summary-card">
            <div className="panel-section-title">Grounding route</div>
            <div className="source-summary-value">{routeLabel(routeInfo)}</div>
            <div className="panel-helper-text">
              {routeInfo?.reason
                ? `Reason: ${humanize(routeInfo.reason)}`
                : 'Ask a question to see how the assistant chose its answer path.'}
            </div>
            <div className="source-summary-stats">
              <span className="source-type-pill"><Quote size={12} /> {sortedCitations.length} total</span>
              <span className="source-type-pill"><Globe2 size={12} /> {webCount} web</span>
              <span className="source-type-pill"><BookOpen size={12} /> {knowledgeCount} knowledge</span>
            </div>
            {researchInfo?.attempted ? (
              <div className="panel-helper-text">
                Research attempted
                {typeof researchInfo.count === 'number' ? ` across ${researchInfo.count} source${researchInfo.count === 1 ? '' : 's'}` : ''}.
              </div>
            ) : null}
            {researchInfo?.error ? <div className="panel-warning">{researchInfo.error}</div> : null}
          </div>

          <div className="panel-section-row">
            <div className="panel-section-title">Evidence</div>
            {sortedCitations.length > 0 ? (
              <button className="ghost-button" type="button" onClick={handleCopySources}>
                {copied ? <Check size={14} /> : <Copy size={14} />}
                {copied ? 'Copied' : 'Copy sources'}
              </button>
            ) : null}
          </div>

          <div className="mode-row wrap-wide" role="group" aria-label="Source filter">
            {(['all', 'web', 'knowledge'] as SourceFilter[]).map((filter) => (
              <button
                key={filter}
                className={`mode-pill ${sourceFilter === filter ? 'active' : ''}`}
                type="button"
                onClick={() => setSourceFilter(filter)}
                aria-pressed={sourceFilter === filter}
              >
                {filter}
              </button>
            ))}
          </div>

          {visibleCitations.length === 0 ? (
            <div className="empty-side-card">
              <BookOpen size={18} />
              <div>No sources selected yet. Ask a question or inspect an assistant message.</div>
            </div>
          ) : null}

          {visibleCitations.map((citation, index) => {
            const score = formatScore(citation.score)
            return (
              <div key={sourceKey(citation, index)} className="source-card">
                <div className="source-card-top">
                  <div className="source-label-row">
                    <span className="source-label">{sourceLabel(citation, index)}</span>
                    <span className="source-type-mini">
                      {citation.source_type === 'web' ? <Globe2 size={12} /> : <BookOpen size={12} />}
                      {citation.source_type === 'web' ? 'web' : 'knowledge'}
                    </span>
                  </div>
                  {score ? <span className="source-score">score {score}</span> : null}
                </div>
                <div className="source-title">{sourceTitle(citation)}</div>
                {sourceMeta(citation) ? <div className="source-meta">{sourceMeta(citation)}</div> : null}
                {normalizeText(citation.snippet) ? <div className="source-snippet">{normalizeText(citation.snippet)}</div> : null}
                {citation.url ? (
                  <a className="source-open-link" href={citation.url} target="_blank" rel="noreferrer noopener">
                    <ExternalLink size={13} />
                    Open source
                  </a>
                ) : null}
              </div>
            )
          })}
        </div>
      ) : (
        <div className="panel-scroll" role="tabpanel" aria-label="Knowledge">
          <div
            className={`upload-card ${dragActive ? 'drag-active' : ''}`}
            onDragOver={(event: DragEvent<HTMLDivElement>) => {
              handleDrag(event)
              setDragActive(true)
            }}
            onDragEnter={(event: DragEvent<HTMLDivElement>) => {
              handleDrag(event)
              setDragActive(true)
            }}
            onDragLeave={(event: DragEvent<HTMLDivElement>) => {
              handleDrag(event)
              setDragActive(false)
            }}
            onDrop={(event: DragEvent<HTMLDivElement>) => {
              handleDrag(event)
              setDragActive(false)
              uploadFiles(event.dataTransfer.files)
            }}
          >
            <div>
              <div className="panel-section-title">Add knowledge</div>
              <div className="panel-helper-text">
                Upload PDFs, DOCX, markdown, text, JSON, CSV, or HTML files to make answers more specific and less generic.
              </div>
              <div className="panel-helper-text">
                {dragActive ? 'Drop files here to add them.' : 'You can also drag and drop files into this panel.'}
              </div>
            </div>
            <button className="primary-button" type="button" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
              <FileUp size={16} />
              {uploading ? 'Uploading...' : 'Upload files'}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              hidden
              multiple
              accept={ACCEPTED_FILE_TYPES}
              onChange={(event: ChangeEvent<HTMLInputElement>) => {
                uploadFiles(event.target.files)
                event.currentTarget.value = ''
              }}
            />
          </div>

          <div className="knowledge-toolbar">
            <div className="search-field-wrap">
              <Search size={14} />
              <input
                className="search-field"
                value={query}
                onChange={(event: ChangeEvent<HTMLInputElement>) => updateQuery(event.target.value)}
                onKeyDown={handleKnowledgeKeyDown}
                placeholder="Search uploaded knowledge"
                aria-label="Search uploaded knowledge"
              />
              {query ? (
                <button className="ghost-icon-button" type="button" onClick={() => updateQuery('')} aria-label="Clear knowledge search">
                  <X size={13} />
                </button>
              ) : null}
            </div>
            <button className="ghost-button" type="button" onClick={runSearch} disabled={!canSearch}>
              Search
            </button>
            <button className="ghost-button" type="button" onClick={onReindex} disabled={uploading || searching || documents.length === 0}>
              <RotateCcw size={14} />
              Reindex
            </button>
          </div>

          {searchResponse ? (
            <div className="search-results-block">
              <div className="panel-section-title">Search results</div>
              <div className="panel-helper-text">
                {searchResponse.count} result{searchResponse.count === 1 ? '' : 's'}
                {typeof searchResponse.confidence === 'number' ? ` | confidence ${searchResponse.confidence.toFixed(2)}` : ''}
              </div>

              {searchResponse.count === 0 ? (
                <div className="empty-side-card compact">
                  <Search size={18} />
                  <div>No matching chunks found. Try fewer keywords or broader wording.</div>
                </div>
              ) : null}

              {searchResponse.results.map((result, index) => {
                const richResult = result as RichCitation
                const score = formatScore(richResult.score)
                return (
                  <div key={sourceKey(richResult, index)} className="source-card compact">
                    <div className="source-card-top">
                      <span className="source-label">{sourceLabel(richResult, index)}</span>
                      {score ? <span className="source-score">score {score}</span> : null}
                    </div>
                    <div className="source-title">{sourceTitle(richResult)}</div>
                    {sourceMeta(richResult) ? <div className="source-meta">{sourceMeta(richResult)}</div> : null}
                    {normalizeText(richResult.snippet) ? <div className="source-snippet">{normalizeText(richResult.snippet)}</div> : null}
                  </div>
                )
              })}
            </div>
          ) : null}

          <div className="panel-section-row">
            <div className="panel-section-title">Documents</div>
            <div className="panel-helper-text">
              {documents.length} document{documents.length === 1 ? '' : 's'} | {totalChunks} chunk{totalChunks === 1 ? '' : 's'} indexed
            </div>
          </div>

          {documents.length > 0 ? (
            <div className="search-field-wrap">
              <Search size={14} />
              <input
                className="search-field"
                value={documentFilter}
                onChange={(event: ChangeEvent<HTMLInputElement>) => updateDocumentFilter(event.target.value)}
                placeholder="Filter documents"
                aria-label="Filter documents"
              />
              {documentFilter ? (
                <button className="ghost-icon-button" type="button" onClick={() => updateDocumentFilter('')} aria-label="Clear document filter">
                  <X size={13} />
                </button>
              ) : null}
            </div>
          ) : null}

          <div className="doc-list">
            {sortedDocuments.length === 0 ? (
              <div className="empty-side-card">
                <Database size={18} />
                <div>No documents yet. Upload PDFs, DOCX, markdown, text, JSON, CSV, or HTML files.</div>
              </div>
            ) : null}

            {sortedDocuments.map((document) => (
              <div key={document.id} className="doc-item">
                <div className="doc-copy">
                  <div className="doc-title">{documentLabel(document)}</div>
                  <div className="doc-meta">
                    {document.chunk_count || 0} chunk{document.chunk_count === 1 ? '' : 's'}
                    {document.source_name ? ` | ${document.source_name}` : ''}
                  </div>
                  {document.text_preview ? <div className="doc-preview">{document.text_preview}</div> : null}
                </div>
                <button
                  className="ghost-icon-button"
                  type="button"
                  onClick={() => {
                    const ok = typeof window === 'undefined' || window.confirm(`Delete \"${documentLabel(document)}\"?`)
                    if (ok) onDeleteDocument(document.id)
                  }}
                  aria-label={`Delete ${documentLabel(document)}`}
                  title="Delete document"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </aside>
  )
}
