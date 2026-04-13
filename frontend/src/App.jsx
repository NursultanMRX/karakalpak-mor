import { useState, useCallback, useRef } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'https://api.kkgrammar.uz'
const API_KEY  = import.meta.env.VITE_API_KEY  || ''

const POS_LABELS = {
  ALM:     'Almastıq',
  ARA_SZ:  'Aralas Sóz',
  ATLQ:    'Atlıq Esim',
  DEM:     'Demonstrativ',
  FYL:     'Feyil',
  JLG:     'Jalǵaw',
  JRD_FYL: 'Járdemshi Feyil',
  KBT:     'Kómekshi Bet',
  RWS:     'Rawısh',
  SNQ:     'Sanaq',
  SYM:     'Sımvol',
  TNS:     'Tańırqaw Sóz',
  TRK:     'Tirkemes',
}

const POS_COLORS = {
  ALM:     '#4f46e5', ARA_SZ:  '#0891b2', ATLQ:    '#059669',
  DEM:     '#7c3aed', FYL:     '#dc2626', JLG:     '#d97706',
  JRD_FYL: '#b45309', KBT:     '#6b7280', RWS:     '#0284c7',
  SNQ:     '#be185d', SYM:     '#374151', TNS:     '#9333ea',
  TRK:     '#047857',
}

const EXAMPLES = [
  'Men mektepke baraman.',
  'Ol kitap oqıydı. Biz universitet studentlerimiz.',
  'Karakalpaqstan RespublikasıÓzbekstan quraмında.',
]

export default function App() {
  const [text, setText]       = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [selected, setSelected] = useState(null)
  const textareaRef = useRef(null)

  const analyze = useCallback(async (input) => {
    const query = (input ?? text).trim()
    if (!query) return
    setLoading(true)
    setError(null)
    setResults(null)
    setSelected(null)
    try {
      const headers = { 'Content-Type': 'application/json' }
      if (API_KEY) headers['X-API-Key'] = API_KEY
      const { data } = await axios.post(
        `${API_URL}/words?morph_fields=all&include_pos_name=true&lemma_fallback=word`,
        { text: query },
        { headers, timeout: 30000 }
      )
      setResults(data)
    } catch (e) {
      if (e.response?.status === 401) setError('Invalid API key.')
      else if (e.response?.status === 429) setError('Rate limit reached. Try again in a moment.')
      else setError(e.response?.data?.detail || e.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }, [text])

  const handleKey = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) analyze()
  }

  const clear = () => { setText(''); setResults(null); setError(null); setSelected(null) }

  const loadExample = (ex) => { setText(ex); setResults(null); setError(null) }

  return (
    <div className="layout">
      {/* ── Header ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="brand-icon">KK</div>
          <span className="brand-name">KKGrammar</span>
        </div>
        <div className="topbar-sub">Karakalpaq tili tahlilshısı</div>
      </header>

      {/* ── Main split pane ── */}
      <div className="split-wrap">
      <div className="split">
        {/* LEFT — input */}
        <div className="pane pane-left">
          <div className="pane-header">
            <span className="pane-lang">Qaraqalpaqsha</span>
            <button className="btn-clear" onClick={clear} title="Clear">✕</button>
          </div>

          <textarea
            ref={textareaRef}
            className="main-textarea"
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Mátinди bul jerge jazıń..."
            disabled={loading}
            spellCheck={false}
          />

          <div className="pane-footer">
            <div className="examples">
              {EXAMPLES.map((ex, i) => (
                <button key={i} className="example-chip" onClick={() => loadExample(ex)}>
                  {ex.length > 32 ? ex.slice(0, 32) + '…' : ex}
                </button>
              ))}
            </div>
            <div className="footer-actions">
              <span className="char-count">{text.length} chars</span>
              <button
                className="btn-analyze"
                onClick={() => analyze()}
                disabled={loading || !text.trim()}
              >
                {loading ? <><span className="spin" />Analyzing…</> : 'Analyze →'}
              </button>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="divider" />

        {/* RIGHT — results */}
        <div className="pane pane-right">
          <div className="pane-header">
            <span className="pane-lang">Tahlil nátijesi</span>
            {results && (
              <div className="result-stats">
                <span className="stat-chip">{results.word_count} words</span>
                <span className="stat-chip">{results.sentence_count} sentences</span>
              </div>
            )}
          </div>

          <div className="result-body">
            {!results && !loading && !error && (
              <div className="placeholder">
                <div className="placeholder-icon">◈</div>
                <p>Analysis results will appear here</p>
                <p className="placeholder-hint">Ctrl+Enter to analyze</p>
              </div>
            )}

            {loading && (
              <div className="placeholder">
                <div className="loading-spinner" />
                <p>Analyzing…</p>
              </div>
            )}

            {error && (
              <div className="error-banner">
                <span>⚠</span> {error}
              </div>
            )}

            {results && (
              <>
                {/* Word chips — click to select */}
                <div className="word-chips">
                  {results.words?.map((w, i) => (
                    <button
                      key={i}
                      className={`word-chip ${selected === i ? 'active' : ''}`}
                      style={{ '--pos-color': POS_COLORS[w.pos] || '#667eea' }}
                      onClick={() => setSelected(selected === i ? null : i)}
                    >
                      <span className="chip-word">{w.word}</span>
                      <span className="chip-pos">{w.pos}</span>
                    </button>
                  ))}
                </div>

                {/* Detail panel for selected word */}
                {selected !== null && results.words[selected] && (
                  <div className="detail-panel">
                    {(() => {
                      const w = results.words[selected]
                      const morphEntries = Object.entries(w.morph || {}).filter(([,v]) => v && v !== '-')
                      return (
                        <>
                          <div className="detail-header">
                            <span className="detail-word">{w.word}</span>
                            <span className="detail-pos-badge" style={{ background: POS_COLORS[w.pos] || '#667eea' }}>
                              {w.pos} · {w.pos_name || POS_LABELS[w.pos] || w.pos}
                            </span>
                          </div>
                          <div className="detail-rows">
                            <div className="detail-row">
                              <span className="detail-key">Lemma</span>
                              <span className="detail-val">{w.lemma || '—'}</span>
                            </div>
                            <div className="detail-row">
                              <span className="detail-key">Sentence</span>
                              <span className="detail-val">#{w.sentence_index + 1}, word #{w.word_index + 1}</span>
                            </div>
                          </div>
                          {morphEntries.length > 0 && (
                            <div className="morph-grid">
                              {morphEntries.map(([k, v]) => (
                                <div key={k} className="morph-cell">
                                  <div className="morph-k">{k}</div>
                                  <div className="morph-v">{v}</div>
                                </div>
                              ))}
                            </div>
                          )}
                        </>
                      )
                    })()}
                  </div>
                )}

                {selected === null && (
                  <p className="click-hint">Click any word to see full morphological analysis</p>
                )}
              </>
            )}
          </div>
        </div>
      </div>
      </div>
    </div>
  )
}
