import { useState, useCallback, useRef } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'https://api.kkgrammar.uz'
const API_KEY  = import.meta.env.VITE_API_KEY  || ''

const POS_LABELS = {
  ALM:     'Almastıq',
  ARA_SZ:  'Aralas Sóz',
  ATLQ:    'Atlıq Esim',
  DEM:     'Kórsetkish Almasıq',
  FYL:     'Feyil',
  JLG:     'Jalǵaw',
  JRD_FYL: 'Járdemshi Feyil',
  KBT:     'Kómekshi Bet',
  RWS:     'Rawısh',
  SNQ:     'Sanaq Esim',
  SYM:     'Belgili Sóz',
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

export default function App() {
  const [text, setText]         = useState('')
  const [results, setResults]   = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [selected, setSelected] = useState(null) // { sentenceIdx, wordIdx }
  const resultsRef = useRef(null)

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
      // Scroll to results on mobile
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    } catch (e) {
      if (e.response?.status === 401) setError('API açqışı nátúrıs. Administratorǵa baylanısıń.')
      else if (e.response?.status === 429) setError('Sorawnlar sánı asıp ketti. Bir ázden keyin qayta urınıp kóriń.')
      else setError(e.response?.data?.detail || e.message || 'Mátin tallawdan ótkerip bolmadı.')
    } finally {
      setLoading(false)
    }
  }, [text])

  const handleKey = (e) => { if (e.key === 'Enter' && e.ctrlKey) analyze() }
  const clear = () => { setText(''); setResults(null); setError(null); setSelected(null) }
  const loadExample = (ex) => { setText(ex); setResults(null); setError(null); setSelected(null) }

  const handleWordClick = (sentenceIdx, wordIdx) => {
    if (selected?.sentenceIdx === sentenceIdx && selected?.wordIdx === wordIdx) {
      setSelected(null)
    } else {
      setSelected({ sentenceIdx, wordIdx })
    }
  }

  // Group words by sentence_index
  const sentences = results ? groupBySentence(results.words) : []

  return (
    <div className="layout">

      {/* ── Header ── */}
      <header className="header">
        <div className="brand">
          <div className="brand-icon">KK</div>
          <span className="brand-name">KKGrammar</span>
        </div>
        <span className="brand-sub">Qaraqalpaq tili grammatikalıq tallaw</span>
      </header>

      {/* ── Content ── */}
      <div className="main-content">

        {/* Input card */}
        <div className="card">
          <div className="card-label">Mátin kiritiń</div>
          <textarea
            className="main-textarea"
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Qaraqalpaqsha mátinińizdi usı jerge jazıń..."
            disabled={loading}
            spellCheck={false}
          />
          <div className="card-footer">
            <div className="examples-label">Mısallar:</div>
            <div className="examples">
              {EXAMPLES.map((ex, i) => (
                <button key={i} className="example-chip" onClick={() => loadExample(ex)}>
                  {ex.length > 40 ? ex.slice(0, 40) + '…' : ex}
                </button>
              ))}
            </div>
            <div className="footer-actions">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span className="char-count">{text.length} harp</span>
                {text && (
                  <button className="btn-clear" onClick={clear} title="Tazalaw">✕</button>
                )}
              </div>
              <button
                className="btn-analyze"
                onClick={() => analyze()}
                disabled={loading || !text.trim()}
              >
                {loading
                  ? <><span className="spin" /> Tallanıp atır…</>
                  : 'Tallawǵa jiberiw →'}
              </button>
            </div>
          </div>
        </div>

        {/* Results card */}
        {(results || loading || error) && (
          <div className="card" ref={resultsRef}>
            <div className="results-header">
              <span className="results-title">Grammatikalıq tallaw nátiyjeleri</span>
              {results && (
                <div className="result-stats">
                  <span className="stat-chip">{results.word_count} sóz</span>
                  <span className="stat-chip">{results.sentence_count} sóylem</span>
                </div>
              )}
            </div>

            {loading && (
              <div className="placeholder">
                <div className="loading-spinner" />
                <p>Tallanıp atır…</p>
              </div>
            )}

            {error && (
              <div className="error-banner">⚠ {error}</div>
            )}

            {results && sentences.map((words, sIdx) => (
              <div key={sIdx}>
                {/* Sentence row */}
                <div className="sentence-block">
                  <div className="sentence-label">Sóylem #{sIdx + 1}</div>
                  <div className="sentence-words">
                    {words.map((w, wIdx) => {
                      const isActive = selected?.sentenceIdx === sIdx && selected?.wordIdx === wIdx
                      return (
                        <div
                          key={wIdx}
                          className={`word-unit ${isActive ? 'active' : ''}`}
                          style={{ '--pos-color': POS_COLORS[w.pos] || '#78716c' }}
                          onClick={() => handleWordClick(sIdx, wIdx)}
                        >
                          <span className="word-text">{w.word}</span>
                          <span className="word-pos-label">{w.pos}</span>
                        </div>
                      )
                    })}
                    <span className="word-punct">.</span>
                  </div>
                </div>

                {/* Detail panel — shown below the sentence that contains the selected word */}
                {selected?.sentenceIdx === sIdx && (() => {
                  const w = words[selected.wordIdx]
                  if (!w) return null
                  const morphEntries = Object.entries(w.morph || {}).filter(([, v]) => v && v !== '-')
                  return (
                    <div className="detail-panel">
                      <div className="detail-header">
                        <span className="detail-word">{w.word}</span>
                        <span
                          className="detail-pos-badge"
                          style={{ background: POS_COLORS[w.pos] || '#78716c' }}
                        >
                          {w.pos} — {w.pos_name || POS_LABELS[w.pos] || w.pos}
                        </span>
                      </div>
                      <div className="detail-rows">
                        <div className="detail-row">
                          <span className="detail-key">Lemma</span>
                          <span className="detail-val">{w.lemma || '—'}</span>
                        </div>
                        <div className="detail-row">
                          <span className="detail-key">Orın</span>
                          <span className="detail-val">{sIdx + 1}-sóylem, {w.word_index + 1}-sóz</span>
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
              </div>
            ))}

            {results && selected === null && (
              <p className="click-hint">Morfologiyalıq belgilerdi kóriw ushın sózge basıń</p>
            )}
          </div>
        )}

        {/* Initial placeholder */}
        {!results && !loading && !error && (
          <div className="card">
            <div className="placeholder">
              <div className="placeholder-icon">◈</div>
              <p>Nátiyje usı jerde kórsetiledi</p>
              <p className="placeholder-hint">Mátin jazıp «Tallawǵa jiberiw» basıń</p>
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

function groupBySentence(words) {
  if (!words?.length) return []
  const map = {}
  for (const w of words) {
    const si = w.sentence_index ?? 0
    if (!map[si]) map[si] = []
    map[si].push(w)
  }
  const keys = Object.keys(map).map(Number).sort((a, b) => a - b)
  return keys.map(k => map[k])
}
