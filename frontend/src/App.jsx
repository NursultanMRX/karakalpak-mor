import { useState, useCallback } from 'react'
import axios from 'axios'

const POS_LABELS = {
  'N': 'Noun',
  'V': 'Verb',
  'ADJ': 'Adjective',
  'ADV': 'Adverb',
  'PRON': 'Pronoun',
  'DET': 'Determiner',
  'ADP': 'Adposition',
  'CONJ': 'Conjunction',
  'INTJ': 'Interjection',
  'NUM': 'Numeral',
  'PART': 'Particle'
}

function App() {
  const [text, setText] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiUrl, setApiUrl] = useState('http://localhost:8000')
  const [apiKey, setApiKey] = useState('')

  const analyzeText = useCallback(async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze')
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const headers = {
        'Content-Type': 'application/json',
      }
      
      if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`
      }

      const response = await axios.post(
        `${apiUrl}/api/analyze`,
        { text: text.trim() },
        { headers }
      )

      setResults(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }, [text, apiUrl, apiKey])

  const clearAll = () => {
    setText('')
    setResults(null)
    setError(null)
  }

  const loadExample = () => {
    setText('Мен мектепке бараман. Бүгин ҳаўа райы жақсы.')
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>🔤 Karakalpak POS & Morphological Analyzer</h1>
        <p>Advanced NLP tool for Karakalpak language analysis</p>
      </header>

      <main className="main-content">
        <section className="input-section">
          <label htmlFor="text-input">Enter Karakalpak Text:</label>
          <div className="textarea-container">
            <textarea
              id="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Мәтінди бул жерге киргизиң..."
              disabled={loading}
            />
          </div>
          <div className="button-group">
            <button 
              className="btn btn-primary" 
              onClick={analyzeText}
              disabled={loading || !text.trim()}
            >
              {loading ? '⏳ Analyzing...' : '🚀 Analyze Text'}
            </button>
            <button 
              className="btn btn-secondary" 
              onClick={loadExample}
              disabled={loading}
            >
              📝 Load Example
            </button>
            <button 
              className="btn btn-secondary" 
              onClick={clearAll}
              disabled={loading}
            >
              🗑️ Clear
            </button>
          </div>
        </section>

        <section className="settings-section">
          <h3>⚙️ API Settings</h3>
          <div className="setting-item">
            <label htmlFor="api-url">API URL:</label>
            <input
              type="text"
              id="api-url"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
              disabled={loading}
            />
          </div>
          <div className="setting-item">
            <label htmlFor="api-key">API Key (optional):</label>
            <input
              type="password"
              id="api-key"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Your API key"
              disabled={loading}
            />
          </div>
        </section>

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Analyzing your text...</p>
          </div>
        )}

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {results && (
          <section className="results-section">
            <h2>📊 Analysis Results</h2>
            
            <div className="stats-bar">
              <div className="stat-item">
                <div className="stat-value">{results.tokens?.length || 0}</div>
                <div className="stat-label">Tokens</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{results.processing_time_ms?.toFixed(2) || '0'}</div>
                <div className="stat-label">Time (ms)</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{results.model_info?.model_name || 'N/A'}</div>
                <div className="stat-label">Model</div>
              </div>
            </div>

            <div className="token-grid">
              {results.tokens?.map((token, index) => (
                <div key={index} className="token-card">
                  <div className="token-header">
                    <span className="token-text">{token.text}</span>
                    <span className="token-pos">
                      {POS_LABELS[token.pos] || token.pos}
                    </span>
                  </div>
                  <div className="token-details">
                    <div className="detail-item">
                      <div className="detail-label">Lemma</div>
                      <div className="detail-value">{token.lemma || '—'}</div>
                    </div>
                    <div className="detail-item">
                      <div className="detail-label">Morphology</div>
                      <div className="detail-value">
                        {token.morphology && Object.keys(token.morphology).length > 0
                          ? Object.entries(token.morphology)
                              .map(([key, value]) => `${key}: ${value}`)
                              .join(', ')
                          : '—'}
                      </div>
                    </div>
                    <div className="detail-item">
                      <div className="detail-label">Confidence</div>
                      <div className="detail-value">
                        {token.confidence ? `${(token.confidence * 100).toFixed(1)}%` : 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  )
}

export default App
