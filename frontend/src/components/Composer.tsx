import { SearchCheck, SendHorizontal } from 'lucide-react'
import { useState } from 'react'
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
  { value: 'auto', label: 'Auto research', hint: 'Use research only when needed' },
  { value: 'off', label: 'Knowledge only', hint: 'Use uploaded knowledge only' },
  { value: 'force', label: 'Force research', hint: 'Always research outside sources' }
]

export default function Composer({ mode, researchMode, busy, setMode, setResearchMode, onSend }: ComposerProps) {
  const [value, setValue] = useState('')

  function submit() {
    const text = value.trim()
    if (!text || busy) return
    onSend(text)
    setValue('')
  }

  return (
    <div className="composer panel">
      <div className="composer-topbar">
        <div className="mode-block">
          <div className="composer-label">Answer style</div>
          <div className="mode-row">
            {modes.map((item) => (
              <button
                key={item}
                className={`mode-pill ${mode === item ? 'active' : ''}`}
                onClick={() => setMode(item)}
                type="button"
              >
                {item}
              </button>
            ))}
          </div>
        </div>

        <div className="mode-block">
          <div className="composer-label">Grounding strategy</div>
          <div className="mode-row wrap-wide">
            {researchModes.map((item) => (
              <button
                key={item.value}
                className={`mode-pill ${researchMode === item.value ? 'active' : ''}`}
                onClick={() => setResearchMode(item.value)}
                type="button"
                title={item.hint}
              >
                <SearchCheck size={14} />
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="composer-row">
        <textarea
          className="composer-textarea"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              submit()
            }
          }}
          placeholder="Ask for analysis, troubleshooting, research, plans, comparisons, or grounded answers from your knowledge base..."
          rows={5}
        />
        <button className="send-button" disabled={busy || !value.trim()} onClick={submit}>
          <SendHorizontal size={16} />
          {busy ? 'Working...' : 'Send'}
        </button>
      </div>
      <div className="composer-hint">Enter sends. Shift+Enter makes a new line. Research mode can combine your uploaded knowledge with web sources.</div>
    </div>
  )
}
