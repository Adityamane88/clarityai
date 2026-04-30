import { RotateCcw, SearchCheck, SendHorizontal, Sparkles, X } from 'lucide-react'
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
  type KeyboardEvent
} from 'react'
import type { ChatMode, ResearchMode } from '../types'

interface ComposerProps {
  mode: ChatMode
  researchMode: ResearchMode
  busy: boolean
  setMode: (mode: ChatMode) => void
  setResearchMode: (mode: ResearchMode) => void
  onSend: (text: string) => void
}

const modes: ChatMode[] = ['concise', 'balanced', 'deep']
const researchModes: { value: ResearchMode; label: string; hint: string }[] = [
  { value: 'auto', label: 'Auto research', hint: 'Use research only when it improves the answer' },
  { value: 'off', label: 'Knowledge only', hint: 'Use uploaded knowledge and conversation context only' },
  { value: 'force', label: 'Force research', hint: 'Always search outside sources before answering' }
]

const starters = [
  { label: 'Summarize', text: 'Summarize the main points, then tell me the key takeaway and the best next step.' },
  { label: 'Compare', text: 'Compare the strongest options, explain the tradeoffs, and end with a clear recommendation.' },
  { label: 'Troubleshoot', text: 'Help me troubleshoot this systematically. Start with the most likely causes and the fastest checks.' },
  { label: 'Plan', text: 'Turn this into a concrete plan with steps, priorities, risks, and the best first action.' }
]

const DRAFT_STORAGE_KEY = 'clarityai:composer:draft:elite'
const MIN_TEXTAREA_HEIGHT = 140
const MAX_TEXTAREA_HEIGHT = 320
const HARD_CHARACTER_LIMIT = 12000
const SOFT_CHARACTER_WARNING = 6000

function normalizeDraft(value: string): string {
  return value
    .replace(/\r\n/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
}

function getInitialDraft(): string {
  if (typeof window === 'undefined') return ''
  try {
    return window.localStorage.getItem(DRAFT_STORAGE_KEY) || ''
  } catch {
    return ''
  }
}

function nextResearchMode(current: ResearchMode): ResearchMode {
  if (current === 'auto') return 'off'
  if (current === 'off') return 'force'
  return 'auto'
}

export default function Composer({ mode, researchMode, busy, setMode, setResearchMode, onSend }: ComposerProps) {
  const [value, setValue] = useState(getInitialDraft)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const isComposingRef = useRef(false)

  const placeholder = useMemo(() => {
    if (researchMode === 'off') {
      return 'Ask grounded questions from your uploaded files, policies, notes, or manuals...'
    }
    if (researchMode === 'force') {
      return 'Ask for current facts, external comparisons, web-backed summaries, or source-grounded research...'
    }
    return 'Ask for analysis, troubleshooting, research, plans, comparisons, or grounded answers from your knowledge base...'
  }, [researchMode])

  const activeResearchHint = useMemo(() => {
    return researchModes.find((item) => item.value === researchMode)?.hint || ''
  }, [researchMode])

  const normalizedValue = normalizeDraft(value)
  const trimmedValue = normalizedValue.trim()
  const canSend = Boolean(trimmedValue) && !busy
  const showLengthWarning = value.length > SOFT_CHARACTER_WARNING

  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      if (value) {
        window.localStorage.setItem(DRAFT_STORAGE_KEY, value)
      } else {
        window.localStorage.removeItem(DRAFT_STORAGE_KEY)
      }
    } catch {
      // Ignore localStorage failures; draft saving is best effort.
    }
  }, [value])

  useEffect(() => {
    const node = textareaRef.current
    if (!node) return

    node.style.height = '0px'
    node.style.height = `${Math.min(Math.max(node.scrollHeight, MIN_TEXTAREA_HEIGHT), MAX_TEXTAREA_HEIGHT)}px`
  }, [value])

  function submit() {
    if (!trimmedValue || busy) return
    onSend(trimmedValue)
    setValue('')
  }

  function insertStarter(text: string) {
    setValue((current) => {
      const base = normalizeDraft(current).trim()
      if (!base) return text
      return `${base}\n\n${text}`
    })

    if (typeof window !== 'undefined') {
      window.requestAnimationFrame(() => textareaRef.current?.focus())
    }
  }

  function clearDraft() {
    setValue('')
    textareaRef.current?.focus()
  }

  function handleChange(event: ChangeEvent<HTMLTextAreaElement>) {
    setValue(event.target.value.slice(0, HARD_CHARACTER_LIMIT))
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (isComposingRef.current) return

    if ((event.metaKey || event.ctrlKey) && ['1', '2', '3'].includes(event.key)) {
      event.preventDefault()
      setMode(modes[Number(event.key) - 1])
      return
    }

    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault()
      setResearchMode(nextResearchMode(researchMode))
      return
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <form
      className="composer panel"
      aria-label="Message composer"
      onSubmit={(event: FormEvent<HTMLFormElement>) => {
        event.preventDefault()
        submit()
      }}
    >
      <div className="composer-topbar">
        <div className="mode-block">
          <div className="composer-label">Answer style</div>
          <div className="mode-row" role="group" aria-label="Answer style">
            {modes.map((item) => (
              <button
                key={item}
                className={`mode-pill ${mode === item ? 'active' : ''}`}
                onClick={() => setMode(item)}
                type="button"
                aria-pressed={mode === item}
                disabled={busy}
                title={`Use ${item} mode`}
              >
                {item}
              </button>
            ))}
          </div>
        </div>

        <div className="mode-block">
          <div className="composer-label">Grounding strategy</div>
          <div className="mode-row wrap-wide" role="group" aria-label="Grounding strategy">
            {researchModes.map((item) => (
              <button
                key={item.value}
                className={`mode-pill ${researchMode === item.value ? 'active' : ''}`}
                onClick={() => setResearchMode(item.value)}
                type="button"
                title={item.hint}
                aria-pressed={researchMode === item.value}
                disabled={busy}
              >
                <SearchCheck size={14} />
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="composer-quickstart-row" aria-label="Prompt helpers">
        {starters.map((starter) => (
          <button
            key={starter.label}
            className="mode-pill subtle"
            type="button"
            onClick={() => insertStarter(starter.text)}
            title={starter.text}
            disabled={busy}
          >
            <Sparkles size={13} />
            {starter.label}
          </button>
        ))}
      </div>

      <div className="composer-row">
        <textarea
          ref={textareaRef}
          className="composer-textarea"
          value={value}
          onChange={handleChange}
          onCompositionStart={() => {
            isComposingRef.current = true
          }}
          onCompositionEnd={() => {
            isComposingRef.current = false
          }}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={5}
          aria-label="Ask a question"
          aria-busy={busy}
          spellCheck
          autoCapitalize="sentences"
        />

        <div className="composer-actions">
          {value ? (
            <button
              className="ghost-button"
              type="button"
              onClick={clearDraft}
              disabled={busy}
              aria-label="Clear draft"
            >
              <X size={14} />
              Clear
            </button>
          ) : null}

          <button className="send-button" type="submit" disabled={!canSend} aria-label={busy ? 'Generating answer' : 'Send message'}>
            <SendHorizontal size={16} />
            {busy ? 'Working...' : 'Send'}
          </button>
        </div>
      </div>

      <div className="composer-footer">
        <div className="composer-hint">
          Enter sends. Shift+Enter adds a new line. Ctrl/Cmd+1-3 changes depth. Ctrl/Cmd+K cycles research mode.
        </div>
        <div className="composer-meta">
          <span className={showLengthWarning ? 'warn' : undefined}>
            {trimmedValue ? `${trimmedValue.length} characters` : activeResearchHint}
          </span>
          <button className="ghost-button" onClick={clearDraft} type="button" disabled={!value || busy}>
            <RotateCcw size={14} />
            Reset
          </button>
        </div>
      </div>
    </form>
  )
}
