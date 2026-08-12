import { useState } from 'react'
import { formatCurrency } from '../format'

const TYPE_LABELS = {
  buy: 'Compra',
  sell: 'Venta',
  deposit: 'Depósito',
  withdrawal: 'Retiro',
}

const TYPE_BADGE_CLASS = {
  buy: 'badge--buy',
  sell: 'badge--sell',
  deposit: 'badge--buy',
  withdrawal: 'badge--sell',
}

function TransactionsList({ transactions, currency, currencyByTicker, onDelete }) {
  const [deletingId, setDeletingId] = useState(null)

  if (transactions.length === 0) {
    return <p className="empty-state">Todavía no has registrado ninguna transacción.</p>
  }

  const sorted = [...transactions].sort((a, b) => new Date(b.executed_at) - new Date(a.executed_at))

  const handleDelete = async (id) => {
    setDeletingId(id)
    try {
      await onDelete(id)
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="table-scroll">
      <table className="positions-table">
        <thead>
          <tr>
            <th>Fecha</th>
            <th>Ticker</th>
            <th>Tipo</th>
            <th className="num">Cantidad</th>
            <th className="num">Precio</th>
            <th className="num">Total</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((tx) => {
            const isCashMovement = tx.transaction_type === 'deposit' || tx.transaction_type === 'withdrawal'
            // Transactions don't store their own currency (a raw price the user
            // entered) - approximated from the position's live-quoted currency
            // when the ticker is still held, falling back to the portfolio's
            // base currency for fully closed-out tickers and for cash movements
            // (always entered in the portfolio's own base currency, see
            // TransactionForm's docstring).
            const txCurrency = isCashMovement ? currency : (currencyByTicker?.get(tx.ticker) ?? currency)
            return (
              <tr key={tx.id}>
                <td>{new Date(tx.executed_at).toLocaleDateString('es-ES')}</td>
                <td>{isCashMovement ? <em>Liquidez</em> : tx.ticker}</td>
                <td>
                  <span className={`badge ${TYPE_BADGE_CLASS[tx.transaction_type]}`}>
                    {TYPE_LABELS[tx.transaction_type] ?? tx.transaction_type}
                  </span>
                </td>
                <td className="num">{isCashMovement ? '—' : tx.quantity}</td>
                <td className="num">{isCashMovement ? '—' : formatCurrency(tx.price, txCurrency)}</td>
                <td className="num">{formatCurrency(tx.quantity * tx.price + tx.fees, txCurrency)}</td>
                <td className="num">
                  <button
                    type="button"
                    className="icon-button"
                    aria-label={`Eliminar transacción de ${tx.ticker}`}
                    disabled={deletingId === tx.id}
                    onClick={() => handleDelete(tx.id)}
                  >
                    {deletingId === tx.id ? '…' : '✕'}
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default TransactionsList
