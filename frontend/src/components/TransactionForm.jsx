import { useState } from 'react'

const EMPTY_FORM = { ticker: '', transaction_type: 'buy', quantity: '', price: '', executed_at: '' }

function TransactionForm({ onSubmit, submitting }) {
  const [form, setForm] = useState(EMPTY_FORM)
  const [error, setError] = useState(null)

  const update = (field) => (event) => setForm({ ...form, [field]: event.target.value })

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError(null)
    try {
      await onSubmit({
        ticker: form.ticker.trim().toUpperCase(),
        transaction_type: form.transaction_type,
        quantity: Number(form.quantity),
        price: Number(form.price),
        executed_at: form.executed_at ? new Date(form.executed_at).toISOString() : undefined,
      })
      setForm(EMPTY_FORM)
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <form className="transaction-form" onSubmit={handleSubmit}>
      <label className="filter">
        <span>Ticker</span>
        <input required value={form.ticker} onChange={update('ticker')} placeholder="AAPL, SAP.DE, AZN.L…" />
      </label>
      <label className="filter">
        <span>Tipo</span>
        <select value={form.transaction_type} onChange={update('transaction_type')}>
          <option value="buy">Compra</option>
          <option value="sell">Venta</option>
        </select>
      </label>
      <label className="filter">
        <span>Cantidad</span>
        <input required type="number" step="any" min="0" value={form.quantity} onChange={update('quantity')} />
      </label>
      <label className="filter">
        <span>Precio (en la divisa del ticker)</span>
        <input required type="number" step="any" min="0" value={form.price} onChange={update('price')} />
      </label>
      <label className="filter">
        <span>Fecha</span>
        <input type="date" value={form.executed_at} onChange={update('executed_at')} />
      </label>
      <button type="submit" className="button-primary" disabled={submitting}>
        {submitting ? 'Guardando…' : 'Añadir transacción'}
      </button>
      {error && <p className="form-error">{error}</p>}
      <p className="transaction-form__hint">
        También puedes añadir activos europeos u otros mercados - usa el sufijo de la bolsa correspondiente
        (p. ej. <code>.DE</code> Xetra, <code>.PA</code> París, <code>.L</code> Londres, <code>.MC</code> Madrid,
        <code>.MI</code> Milán, <code>.AS</code> Ámsterdam, <code>.SW</code> Suiza) e introduce el precio en la
        divisa en la que cotiza ese ticker - el total de la cartera lo convertimos automáticamente.
      </p>
    </form>
  )
}

export default TransactionForm
