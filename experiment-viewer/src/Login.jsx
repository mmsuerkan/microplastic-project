import { useState } from 'react'

// Sifre kontrolu tarayicida yapilir (statik site, backend yok).
// Bu, siteyi "rastgele/dis arastirma gruplarindan" gizler; banka duzeyi guvenlik degildir.
function Login({ password, onSuccess }) {
  const [input, setInput] = useState('')
  const [error, setError] = useState(false)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (input === password) {
      onSuccess()
    } else {
      setError(true)
    }
  }

  return (
    <div className="login-screen">
      <form className="login-box" onSubmit={handleSubmit}>
        <div className="login-logo">
          <span className="logo-icon">◉</span>
          <span className="logo-text">MicroTrack</span>
        </div>
        <p className="login-subtitle">Mikroplastik Deney Analiz Sistemi</p>
        <input
          type="password"
          className={'login-input' + (error ? ' error' : '')}
          placeholder="Sifre"
          value={input}
          onChange={(e) => { setInput(e.target.value); setError(false) }}
          autoFocus
        />
        {error && <p className="login-error">Hatali sifre, tekrar deneyin.</p>}
        <button type="submit" className="login-btn">Giris</button>
      </form>
    </div>
  )
}

export default Login
