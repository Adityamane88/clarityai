import { BookOpen, Database, ExternalLink, FileUp, Globe2, Quote, RotateCcw, Search, Trash2 } from 'lucide-react'
import { useRef, useState } from 'react'
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
  const [tab, setTab] = useState<'sources' | 'knowledge'>('sources')
  const [query, setQuery] = useState('')
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  return (
    <aside className="right-panel panel">
      <div className="tab-row">
        <button className={`tab-button ${tab === 'sources' ? 'active' : ''}`} onClick={() => setTab('sources')}>
          <Quote size={14} />
          Sources
        </button>
        <button className={`tab-button ${tab === 'knowledge' ? 'active' : ''}`} onClick={() => setTab('knowledge')}>
          <Database size={14} />
          Knowledge
        </button>
      </div>

      {tab === 'sources' ? (
        <div className="panel-scroll">
          <div className="source-summary-card">
            <div className="panel-section-title">Grounding route</div>
            <div className="source-summary-value">{routeInfo?.resolved_route || 'No answer yet'}</div>
            <div className="panel-helper-text">
              {routeInfo?.reason ? `Reason: ${routeInfo.reason.split('_').join(' ')}` : 'Ask a question to see how the assistant routed the answer.'}
            </div>
            {researchInfo?.error ? <div className="panel-warning">{researchInfo.error}</div> : null}
          </div>

          <div className="panel-section-title">Evidence</div>
          {citations.length === 0 ? (
            <div className="empty-side-card">
              <BookOpen size={18} />
              <div>No sources selected yet. Ask a question or inspect an assistant message.</div>
            </div>
          ) : null}
          {citations.map((citation) => (
            <div key={`${citation.id}-${citation.chunk_id ?? citation.document_id}`} className="source-card">
              <div className="source-card-top">
                <div className="source-label-row">
                  <span className="source-label">{citation.label}</span>
                  <span className="source-type-mini">
                    {citation.source_type === 'web' ? <Globe2 size={12} /> : <BookOpen size={12} />}
                    {citation.source_type === 'web' ? 'web' : 'knowledge'}
                  </span>
                </div>
                <span className="source-score">score {citation.score.toFixed(2)}</span>
              </div>
              <div className="source-title">{citation.document_title}</div>
              <div className="source-meta">
                {citation.source_name}
                {citation.page_label ? ` | page ${citation.page_label}` : ''}
                {citation.published_at ? ` | ${citation.published_at}` : ''}
              </div>
              <div className="source-snippet">{citation.snippet}</div>
              {citation.url ? (
                <a className="source-open-link" href={citation.url} target="_blank" rel="noreferrer">
                  <ExternalLink size={13} />
                  Open source
                </a>
              ) : null}
            </div>
          ))}
        </div>
      ) : (
        <div className="panel-scroll">
          <div className="upload-card">
            <div>
              <div className="panel-section-title">Add knowledge</div>
              <div className="panel-helper-text">Upload PDFs, notes, manuals, SOPs, or structured files to make answers more specific and less generic.</div>
            </div>
            <button className="primary-button" onClick={() => fileInputRef.current?.click()}>
              <FileUp size={16} />
              {uploading ? 'Uploading...' : 'Upload file'}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) onUpload(file)
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
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') onSearch(query)
                }}
                placeholder="Search uploaded knowledge"
              />
            </div>
            <button className="ghost-button" onClick={() => onSearch(query)} disabled={searching}>
              Search
            </button>
            <button className="ghost-button" onClick={onReindex}>
              <RotateCcw size={14} />
              Reindex
            </button>
          </div>

          {searchResponse ? (
            <div className="search-results-block">
              <div className="panel-section-title">Search results</div>
              <div className="panel-helper-text">
                {searchResponse.count} result{searchResponse.count === 1 ? '' : 's'} | confidence {searchResponse.confidence.toFixed(2)}
              </div>
              {searchResponse.results.map((result) => (
                <div key={`${result.id}-${result.chunk_id ?? result.document_id}`} className="source-card compact">
                  <div className="source-card-top">
                    <span className="source-label">{result.label}</span>
                    <span className="source-score">score {result.score.toFixed(2)}</span>
                  </div>
                  <div className="source-title">{result.document_title}</div>
                  <div className="source-snippet">{result.snippet}</div>
                </div>
              ))}
            </div>
          ) : null}

          <div className="panel-section-title">Documents</div>
          <div className="doc-list">
            {documents.length === 0 ? (
              <div className="empty-side-card">
                <Database size={18} />
                <div>No documents yet. Upload PDFs, markdown, text, JSON, or CSV files.</div>
              </div>
            ) : null}

            {documents.map((document) => (
              <div key={document.id} className="doc-item">
                <div className="doc-copy">
                  <div className="doc-title">{document.title}</div>
                  <div className="doc-meta">{document.chunk_count} chunks | {document.source_name}</div>
                  <div className="doc-preview">{document.text_preview}</div>
                </div>
                <button className="ghost-icon-button" onClick={() => onDeleteDocument(document.id)} aria-label="Delete document">
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
