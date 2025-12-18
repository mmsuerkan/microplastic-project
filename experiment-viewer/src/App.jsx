import { useState, useEffect } from 'react'
import './App.css'

const S3_BASE_URL = "https://microplastic-experiments.s3.us-east-1.amazonaws.com"

function App() {
  const [experiments, setExperiments] = useState([])
  const [stats, setStats] = useState({ total: 0, success: 0, fail: 0 })
  const [selected, setSelected] = useState(null)
  const [filter, setFilter] = useState({ status: 'all', category: 'all', date: 'all', view: 'all', repeat: 'all', failReason: 'all', search: '' })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [videoLoading, setVideoLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [overrides, setOverrides] = useState(() => {
    const saved = localStorage.getItem('microtrack_overrides')
    return saved ? JSON.parse(saved) : {}
  })
  const [reanalyzeList, setReanalyzeList] = useState(() => {
    const saved = localStorage.getItem('microtrack_reanalyze')
    return saved ? JSON.parse(saved) : {}
  })

  useEffect(() => {
    Promise.all([
      fetch(S3_BASE_URL + '/experiments.json').then(r => r.json()),
      fetch(S3_BASE_URL + '/overrides.json').then(r => r.json()).catch(() => ({})),
      fetch(S3_BASE_URL + '/reanalyze_list.json').then(r => r.json()).catch(() => ({}))
    ]).then(([data, overridesData, reanalyzeData]) => {
      setExperiments(data.experiments || [])
      setStats({ total: data.total || 0, success: data.success || 0, fail: data.fail || 0 })
      const localOverrides = JSON.parse(localStorage.getItem('microtrack_overrides') || '{}')
      const localReanalyze = JSON.parse(localStorage.getItem('microtrack_reanalyze') || '{}')
      setOverrides({ ...overridesData, ...localOverrides })
      setReanalyzeList({ ...reanalyzeData, ...localReanalyze })
      setLoading(false)
    }).catch(err => {
      console.error('Hata:', err)
      setError(err.message)
      setLoading(false)
    })
  }, [])

  useEffect(() => {
    localStorage.setItem('microtrack_overrides', JSON.stringify(overrides))
  }, [overrides])

  useEffect(() => {
    localStorage.setItem('microtrack_reanalyze', JSON.stringify(reanalyzeList))
  }, [reanalyzeList])

  useEffect(() => { if (selected) setVideoLoading(true) }, [selected])

  const getEffectiveStatus = (exp) => {
    if (overrides[exp.id] === 'success') return 'success'
    if (overrides[exp.id] === 'fail') return 'fail'
    return exp.status
  }

  const isOverridden = (exp) => overrides[exp.id] !== undefined
  const isMarkedForReanalyze = (exp) => reanalyzeList[exp.id] !== undefined

  const toggleOverride = (exp) => {
    const newOverrides = { ...overrides }
    if (overrides[exp.id]) { delete newOverrides[exp.id] }
    else { newOverrides[exp.id] = exp.status === 'fail' ? 'success' : 'fail' }
    setOverrides(newOverrides)
  }

  const toggleReanalyze = (exp) => {
    const newList = { ...reanalyzeList }
    if (reanalyzeList[exp.id]) { delete newList[exp.id] }
    else { newList[exp.id] = { path: exp.path, date: exp.date, view: exp.view, repeat: exp.repeat, category: exp.category, code: exp.code, fail_reason: exp.fail_reason, added_at: new Date().toISOString() } }
    setReanalyzeList(newList)
  }

  const exportOverrides = () => {
    const json = JSON.stringify(overrides, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'overrides.json'
    link.click()
    URL.revokeObjectURL(url)
  }

  const exportReanalyzeList = () => {
    const json = JSON.stringify(reanalyzeList, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'reanalyze_list.json'
    link.click()
    URL.revokeObjectURL(url)
  }

  const categories = [...new Set(experiments.map(e => e.category))].sort()
  const dates = [...new Set(experiments.map(e => e.date))].sort()
  const views = [...new Set(experiments.map(e => e.view))].sort()
  const repeats = ['FIRST', 'SECOND', 'THIRD']
  const failReasonsList = [...new Set(experiments.filter(e => e.fail_reason).map(e => e.fail_reason))]

  const filtered = experiments.filter(exp => {
    const effectiveStatus = getEffectiveStatus(exp)
    if (filter.status !== 'all' && effectiveStatus !== filter.status) return false
    if (filter.category !== 'all' && exp.category !== filter.category) return false
    if (filter.date !== 'all' && exp.date !== filter.date) return false
    if (filter.view !== 'all' && exp.view !== filter.view) return false
    if (filter.repeat !== 'all' && exp.repeat !== filter.repeat) return false
    if (filter.failReason !== 'all' && exp.fail_reason !== filter.failReason) return false
    if (filter.search && !exp.code.toLowerCase().includes(filter.search.toLowerCase())) return false
    return true
  })

  const failReasons = experiments.filter(e => e.status === 'fail').reduce((acc, e) => { acc[e.fail_reason] = (acc[e.fail_reason] || 0) + 1; return acc }, {})

  const overrideCount = Object.keys(overrides).length
  const reanalyzeCount = Object.keys(reanalyzeList).length
  const effectiveStats = {
    total: stats.total,
    success: stats.success + Object.values(overrides).filter(v => v === 'success').length - Object.values(overrides).filter(v => v === 'fail').length,
    fail: stats.fail - Object.values(overrides).filter(v => v === 'success').length + Object.values(overrides).filter(v => v === 'fail').length
  }

  const exportToCSV = () => {
    const headers = ['Code', 'Status', 'Effective Status', 'Fail Reason', 'Date', 'View', 'Repeat', 'Category']
    const rows = filtered.map(exp => [exp.code, exp.status, getEffectiveStatus(exp), exp.fail_reason || '', exp.date, exp.view, exp.repeat, exp.category])
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'experiments_' + new Date().toISOString().split('T')[0] + '.csv'
    link.click()
    URL.revokeObjectURL(url)
  }

  if (loading) return (<div className="loading-screen"><div className="loader"></div><p>Veriler yukleniyor...</p></div>)
  if (error) return (<div className="error-screen"><h2>Baglanti Hatasi</h2><p>{error}</p></div>)

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="logo"><span className="logo-icon">◉</span><span className="logo-text">MicroTrack</span></div>
          <span className="header-subtitle">Mikroplastik Deney Analiz Sistemi</span>
        </div>
        <div className="stats-bar">
          <div className="stat-item total"><span className="stat-value">{effectiveStats.total}</span><span className="stat-label">Toplam</span></div>
          <div className="stat-item success"><span className="stat-value">{effectiveStats.success}</span><span className="stat-label">Basarili</span><span className="stat-percent">{effectiveStats.total > 0 ? ((effectiveStats.success/effectiveStats.total)*100).toFixed(0) : 0}%</span></div>
          <div className="stat-item fail"><span className="stat-value">{effectiveStats.fail}</span><span className="stat-label">Basarisiz</span><span className="stat-percent">{effectiveStats.total > 0 ? ((effectiveStats.fail/effectiveStats.total)*100).toFixed(0) : 0}%</span></div>
          {overrideCount > 0 && (<div className="stat-item override"><span className="stat-value">{overrideCount}</span><span className="stat-label">Duzeltme</span><button className="export-override-btn" onClick={exportOverrides} title="Duzeltmeleri indir">↓</button></div>)}
          {reanalyzeCount > 0 && (<div className="stat-item reanalyze"><span className="stat-value">{reanalyzeCount}</span><span className="stat-label">Tekrar Analiz</span><button className="export-reanalyze-btn" onClick={exportReanalyzeList} title="Listeyi indir">↓</button></div>)}
        </div>
      </header>

      <div className="main-content">
        <aside className={"sidebar " + (sidebarOpen ? "open" : "closed")}>
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>{sidebarOpen ? '◀' : '▶'}</button>
          <div className="filters">
            <div className="search-wrapper"><span className="search-icon">⌕</span><input type="text" placeholder="Deney ara..." value={filter.search} onChange={e => setFilter({...filter, search: e.target.value})} className="search-input" /></div>
            <div className="filter-row">
              <select value={filter.status} onChange={e => setFilter({...filter, status: e.target.value})} className="filter-select"><option value="all">Tum Durumlar</option><option value="success">Basarili</option><option value="fail">Basarisiz</option></select>
              <select value={filter.category} onChange={e => setFilter({...filter, category: e.target.value})} className="filter-select"><option value="all">Tum Kategoriler</option>{categories.map(cat => <option key={cat} value={cat}>{cat}</option>)}</select>
            </div>
            <div className="filter-row">
              <select value={filter.date} onChange={e => setFilter({...filter, date: e.target.value})} className="filter-select"><option value="all">Tum Tarihler</option>{dates.map(d => <option key={d} value={d}>{d}</option>)}</select>
              <select value={filter.view} onChange={e => setFilter({...filter, view: e.target.value})} className="filter-select"><option value="all">Tum Gorunumler</option>{views.map(v => <option key={v} value={v}>{v}</option>)}</select>
            </div>
            <div className="filter-row">
              <select value={filter.repeat} onChange={e => setFilter({...filter, repeat: e.target.value})} className="filter-select"><option value="all">Tum Tekrarlar</option>{repeats.map(r => <option key={r} value={r}>{r}</option>)}</select>
              {filter.status === 'fail' && (<select value={filter.failReason} onChange={e => setFilter({...filter, failReason: e.target.value})} className="filter-select"><option value="all">Tum Nedenler</option>{failReasonsList.map(r => <option key={r} value={r}>{r.replace('_', ' ')}</option>)}</select>)}
            </div>
          </div>
          {filter.status === 'fail' && Object.keys(failReasons).length > 0 && (<div className="fail-reasons">{Object.entries(failReasons).map(([reason, count]) => (<span key={reason} className={'reason-badge ' + reason.replace('_', '-') + (filter.failReason === reason ? ' active' : '')} onClick={() => setFilter({...filter, failReason: filter.failReason === reason ? 'all' : reason})} style={{ cursor: 'pointer' }}>{reason.replace('_', ' ')}: {count}</span>))}</div>)}
          <div className="list-header"><span className="list-count">{filtered.length} deney</span><button className="export-btn" onClick={exportToCSV}>CSV Indir</button></div>
          <div className="experiment-list">
            {filtered.map((exp, idx) => (
              <div key={exp.path + idx} className={'experiment-item ' + getEffectiveStatus(exp) + (selected && selected.path === exp.path ? ' selected' : '') + (isOverridden(exp) ? ' overridden' : '')} onClick={() => setSelected(exp)}>
                <div className="exp-status-dot"></div>
                <div className="exp-info">
                  <div className="exp-code-row"><span className="exp-code">{exp.code}</span>{exp.status === 'fail' && !isOverridden(exp) && (<span className="exp-reason">{exp.fail_reason.replace('_', ' ')}</span>)}{isOverridden(exp) && (<span className="override-badge">DUZELTILDI</span>)}</div>
                  <div className="exp-meta"><span>{exp.category}</span><span>{exp.date}</span><span>{exp.view}</span><span className="repeat-tag">{exp.repeat}</span></div>
                </div>
              </div>
            ))}
          </div>
        </aside>

        <main className="detail-panel">
          {selected ? (
            <div className="detail-content">
              <div className="detail-header">
                <div className="header-main">
                  <h2>{selected.code}</h2>
                  <span className="detail-category">{selected.category}</span>
                  <div className={'status-badge ' + getEffectiveStatus(selected)}>{getEffectiveStatus(selected) === 'success' ? 'BASARILI' : selected.fail_reason?.replace('_', ' ').toUpperCase()}</div>
                  {isOverridden(selected) && (<span className="override-indicator">DUZELTILDI</span>)}
                </div>
                <div className="header-info">
                  <span className="info-tag"><b>Tarih:</b> {selected.date}</span>
                  <span className="info-tag"><b>Gorus:</b> {selected.view}</span>
                  <span className="info-tag"><b>Tekrar:</b> {selected.repeat}</span>
                  <button className={'override-btn ' + (isOverridden(selected) ? 'active' : '')} onClick={(e) => { e.stopPropagation(); toggleOverride(selected); }}>{isOverridden(selected) ? '↩ Geri Al' : (selected.status === 'fail' ? '✓ Basarili Isaretle' : '✗ Basarisiz Isaretle')}</button>
                  <button className={'reanalyze-btn ' + (isMarkedForReanalyze(selected) ? 'active' : '')} onClick={(e) => { e.stopPropagation(); toggleReanalyze(selected); }}>{isMarkedForReanalyze(selected) ? '↩ Listeden Cikar' : '🔄 Tekrar Analiz Edilsin'}</button>
                </div>
              </div>
              <div className="detail-grid-2x2">
                <div className="grid-item video-cell">
                  <h3>Video</h3>
                  <div className="video-container">
                    {selected.files && selected.files.includes('output_video.mp4') ? (<>{videoLoading && (<div className="video-loader"><div className="loader"></div><span>Video yukleniyor...</span></div>)}<video key={selected.path} controls autoPlay loop muted playsInline className="video-player" style={{ opacity: videoLoading ? 0 : 1, transition: 'opacity 0.3s' }} onLoadedData={() => setVideoLoading(false)} onError={() => setVideoLoading(false)}><source src={S3_BASE_URL + '/' + selected.path + '/output_video.mp4'} type="video/mp4" /></video></>) : (<div className="no-video">Video mevcut degil</div>)}
                  </div>
                </div>
                <div className="grid-item path-cell">
                  <h3>Takip Yolu</h3>
                  {selected.files && selected.files.includes('auto_tracked_path.jpg') ? (<div className="path-image" onClick={() => window.open(S3_BASE_URL + '/' + selected.path + '/auto_tracked_path.jpg', '_blank')}><img src={S3_BASE_URL + '/' + selected.path + '/auto_tracked_path.jpg'} alt="Takip Yolu" /></div>) : (<div className="no-data">Path resmi yok</div>)}
                </div>
                <div className="grid-item optik-cell">
                  <h3>Optik Akis</h3>
                  {selected.files && selected.files.includes('optical_flow_vectors.png') ? (<div className="optik-image" onClick={() => window.open(S3_BASE_URL + '/' + selected.path + '/optical_flow_vectors.png', '_blank')}><img src={S3_BASE_URL + '/' + selected.path + '/optical_flow_vectors.png'} alt="Optik Akis" /></div>) : (<div className="no-data">Optik akis resmi yok</div>)}
                </div>
                <div className="grid-item metrics-cell">
                  <h3>Olcumler</h3>
                  {selected.metrics ? (<div className="metrics-list">{Object.entries(selected.metrics).map(([key, value]) => (<div key={key} className="metric-row"><span className="metric-label">{key}</span><span className="metric-value">{value}</span></div>))}</div>) : (<p className="no-data">Metrik verisi yok</p>)}
                </div>
              </div>
            </div>
          ) : (<div className="no-selection"><div className="no-selection-icon">◎</div><h2>Deney Secin</h2><p>Soldaki listeden bir deney secerek analiz sonuclarini inceleyin.</p></div>)}
        </main>
      </div>
    </div>
  )
}

export default App
