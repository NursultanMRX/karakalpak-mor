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
  'Karakalpaqstan Respublikası Ózbekstan quramında.',
]

// Key morph features to show on card (without clicking)
const KEY_MORPH = ['Case', 'Number', 'Tense', 'Person', 'Degree']

export default function App() {
  const [text, setText]         = useState('')
  const [results, setResults]   = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
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
      if (e.response?.status === 401) setError('API açqışı nátúrıs. Administratorǵa múrájaat etiń.')
      else if (e.response?.status === 429) setError('Sorawnlar sánı asıp ketti. Bir ázden soń qayta urınıp kóriń.')
      else setError(e.response?.data?.detail || e.message || 'Qayta islew múmkin bolmadı.')
    } finally {
      setLoading(false)
    }
  }, [text])

  const handleKey = (e) => { if (e.key === 'Enter' && e.ctrlKey) analyze() }
  const clear = () => { setText(''); setResults(null); setError(null); setSelected(null) }
  const loadExample = (ex) => { setText(ex); setResults(null); setError(null); setSelected(null) }

  return (
    <div className="layout">

      {/* ── Header ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="brand-icon">KK</div>
          <span className="brand-name">KKGrammar</span>
        </div>
        <div className="topbar-sub">Qaraqalpaq tili grammatika tallaw sisteması</div>
      </header>

      {/* ── Split ── */}
      <div className="split-wrap">
        <div className="split">

          {/* LEFT — input */}
          <div className="pane pane-left">
            <div className="pane-header">
              <span className="pane-lang">Qaraqalpaqsha mátin</span>
              <button className="btn-clear" onClick={clear} title="Tazalaw">✕</button>
            </div>

            <textarea
              ref={textareaRef}
              className="main-textarea"
              value={text}
              onChange={e => setText(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Mátinińizdi usı jerge jazıń..."
              disabled={loading}
              spellCheck={false}
            />

            <div className="pane-footer">
              <div className="examples-label">Mısallar:</div>
              <div className="examples">
                {EXAMPLES.map((ex, i) => (
                  <button key={i} className="example-chip" onClick={() => loadExample(ex)}>
                    {ex.length > 36 ? ex.slice(0, 36) + '…' : ex}
                  </button>
                ))}
              </div>
              <div className="footer-actions">
                <span className="char-count">{text.length} harp</span>
                <button
                  className="btn-analyze"
                  onClick={() => analyze()}
                  disabled={loading || !text.trim()}
                >
                  {loading
                    ? <><span className="spin" /> Tallanıp atır…</>
                    : 'Tallawdan ótkeriw →'}
                </button>
              </div>
              <div className="hint">Ctrl+Enter — tez tallawdan ótkeriw</div>
            </div>
          </div>

          {/* Divider */}
          <div className="divider" />

          {/* RIGHT — results */}
          <div className="pane pane-right">
            <div className="pane-header">
              <span className="pane-lang">Tallawdan ótkeriw nátiyjeleri</span>
              {results && (
                <div className="result-stats">
                  <span className="stat-chip">{results.word_count} sóz</span>
                  <span className="stat-chip">{results.sentence_count} sóylem</span>
                </div>
              )}
            </div>

            <div className="result-body">
              {!results && !loading && !error && (
                <div className="placeholder">
                  <div className="placeholder-icon">◈</div>
                  <p>Nátiyje usı jerde kórsetiledi</p>
                  <p className="placeholder-hint">Mátin jazıp «Tallawdan ótkeriw» basıń</p>
                </div>
              )}

              {loading && (
                <div className="placeholder">
                  <div className="loading-spinner" />
                  <p>Tallanıp atır…</p>
                </div>
              )}

              {error && (
                <div className="error-banner">⚠ {error}</div>
              )}

              {results && (
                <>
                  {/* Word cards */}
                  <div className="word-chips">
                    {results.words?.map((w, i) => {
                      const keyMorph = KEY_MORPH
                        .map(k => w.morph?.[k])
                        .filter(Boolean)
                      return (
                        <button
                          key={i}
                          className={`word-chip ${selected === i ? 'active' : ''}`}
                          style={{ '--pos-color': POS_COLORS[w.pos] || '#667eea' }}
                          onClick={() => setSelected(selected === i ? null : i)}
                        >
                          <span className="chip-word">{w.word}</span>
                          <span className="chip-pos">{w.pos}</span>
                          <span className="chip-pos-name">{POS_LABELS[w.pos] || w.pos}</span>
                          {w.lemma && w.lemma !== w.word && (
                            <span className="chip-lemma">↳ {w.lemma}</span>
                          )}
                          {keyMorph.length > 0 && (
                            <span className="chip-morph">{keyMorph.join(' · ')}</span>
                          )}
                        </button>
                      )
                    })}
                  </div>

                  {/* Detail panel */}
                  {selected !== null && results.words[selected] && (() => {
                    const w = results.words[selected]
                    const morphEntries = Object.entries(w.morph || {}).filter(([, v]) => v && v !== '-')
                    return (
                      <div className="detail-panel">
                        <div className="detail-header">
                          <span className="detail-word">{w.word}</span>
                          <span className="detail-pos-badge" style={{ background: POS_COLORS[w.pos] || '#667eea' }}>
                            {w.pos} — {w.pos_name || POS_LABELS[w.pos] || w.pos}
                          </span>
                        </div>
                        <div className="detail-rows">
                          <div className="detail-row">
                            <span className="detail-key">Lemma</span>
                            <span className="detail-val">{w.lemma || '—'}</span>
                          </div>
                          <div className="detail-row">
                            <span className="detail-key">Sóylem</span>
                            <span className="detail-val">#{w.sentence_index + 1}, {w.word_index + 1}-sóz</span>
                          </div>
                        </div>
                        {morphEntries.length > 0 && (
                          <>
                            <div className="detail-section-title">Morfologiyalıq belgiler</div>
                            <div className="morph-grid">
                              {morphEntries.map(([k, v]) => (
                                <div key={k} className="morph-cell">
                                  <div className="morph-k">{k}</div>
                                  <div className="morph-v">{v}</div>
                                </div>
                              ))}
                            </div>
                          </>
                        )}
                      </div>
                    )
                  })()}

                  {selected === null && (
                    <p className="click-hint">Tolıq morfologiyalıq belgilerdi kóriw ushın sózdi basıń</p>
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
