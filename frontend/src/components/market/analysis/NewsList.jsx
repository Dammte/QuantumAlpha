function formatPublishedAt(value) {
  if (!value) return null
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return null
  return d.toLocaleDateString('es-ES', { day: '2-digit', month: 'short', year: 'numeric' })
}

function NewsList({ news, emptyMessage = 'No hay noticias recientes disponibles para este ticker.' }) {
  if (news.length === 0) {
    return <p className="empty-state">{emptyMessage}</p>
  }

  return (
    <ul className="news-list">
      {news.map((item, index) => (
        <li key={item.link ?? index} className="news-list__item">
          <a href={item.link ?? undefined} target="_blank" rel="noreferrer" className="news-list__title">
            {item.title}
          </a>
          <p className="news-list__meta">
            {item.publisher}
            {formatPublishedAt(item.published_at) && ` · ${formatPublishedAt(item.published_at)}`}
          </p>
        </li>
      ))}
    </ul>
  )
}

export default NewsList
