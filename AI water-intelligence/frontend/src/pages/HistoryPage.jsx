import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, RefreshCw, Trash2, Clock, 
  Database, Tag, Filter, CheckCircle, AlertCircle 
} from 'lucide-react';
import api from '../api';

export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterType, setFilterType] = useState('all');

  useEffect(() => {
    loadHistory();
  }, []);

  async function loadHistory() {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getHistory(50);
      setHistory(data || []);
    } catch (err) {
      setError(err.message || 'Failed to fetch prediction history');
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(id) {
    if (!window.confirm('Delete this prediction record?')) return;
    try {
      await api.deleteHistoryItem(id);
      setHistory(prev => prev.filter(item => item.id !== id));
    } catch (err) {
      alert('Failed to delete: ' + err.message);
    }
  }

  const filteredHistory = filterType === 'all' 
    ? history 
    : history.filter(item => item.prediction_type === filterType || item.mode === filterType);

  return (
    <div className="space-y-6 pb-16">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center space-x-3">
            <Database className="w-8 h-8 text-cyan-400" />
            <span>Database Prediction History</span>
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Persistent audit log of executed flood forecasts and scenario simulations stored in PostgreSQL / SQLAlchemy.
          </p>
        </div>

        <button
          onClick={loadHistory}
          disabled={loading}
          className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh History</span>
        </button>
      </div>

      {/* Filter Tabs */}
      <div className="flex flex-wrap gap-2 text-xs">
        <button
          onClick={() => setFilterType('all')}
          className={`px-3 py-1.5 rounded-lg font-medium transition ${
            filterType === 'all' ? 'bg-cyan-500 text-slate-950 font-bold' : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
          }`}
        >
          All Records ({history.length})
        </button>
        <button
          onClick={() => setFilterType('historical_replay')}
          className={`px-3 py-1.5 rounded-lg font-medium transition ${
            filterType === 'historical_replay' ? 'bg-cyan-500 text-slate-950 font-bold' : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
          }`}
        >
          Historical Replay
        </button>
        <button
          onClick={() => setFilterType('scenario')}
          className={`px-3 py-1.5 rounded-lg font-medium transition ${
            filterType === 'scenario' ? 'bg-cyan-500 text-slate-950 font-bold' : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
          }`}
        >
          Scenario Simulations
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center min-h-[30vh]">
          <div className="w-8 h-8 border-4 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin"></div>
        </div>
      ) : error ? (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-xs">
          {error}
        </div>
      ) : filteredHistory.length === 0 ? (
        <div className="p-12 text-center rounded-2xl bg-slate-900 border border-slate-800 text-slate-400 text-sm">
          No prediction records found. Execute a historical replay or scenario simulation to generate audit logs.
        </div>
      ) : (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 overflow-hidden shadow-xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase font-medium bg-slate-950/60">
                  <th className="py-3 px-4">ID</th>
                  <th className="py-3 px-4">Timestamp</th>
                  <th className="py-3 px-4">Prediction Target</th>
                  <th className="py-3 px-4">Mode</th>
                  <th className="py-3 px-4">Event / Gauge</th>
                  <th className="py-3 px-4 text-right">Predicted</th>
                  <th className="py-3 px-4 text-right">Observed</th>
                  <th className="py-3 px-4 text-right">Residual</th>
                  <th className="py-3 px-4 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {filteredHistory.map((item) => {
                  const dateStr = item.timestamp ? new Date(item.timestamp).toLocaleString() : 'N/A';
                  return (
                    <tr key={item.id} className="hover:bg-slate-800/30">
                      <td className="py-3 px-4 text-slate-500">#{item.id}</td>
                      <td className="py-3 px-4 text-slate-400 text-[11px] whitespace-nowrap">{dateStr}</td>
                      <td className="py-3 px-4 font-sans font-semibold text-white">
                        <span className="capitalize">{item.prediction_type.replace(/_/g, ' ')}</span>
                      </td>
                      <td className="py-3 px-4 font-sans">
                        <span className={`px-2 py-0.5 rounded text-[11px] font-medium ${
                          item.mode === 'historical_replay'
                            ? 'bg-blue-950 text-blue-300 border border-blue-800/60'
                            : 'bg-amber-950 text-amber-300 border border-amber-800/60'
                        }`}>
                          {item.mode === 'historical_replay' ? 'Replay' : 'Scenario'}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-cyan-300">
                        {item.event_id || item.gauge_id || 'Custom Watershed'}
                      </td>
                      <td className="py-3 px-4 text-right font-bold text-cyan-400">
                        {item.predicted_value !== null ? `${item.predicted_value} ${item.unit || ''}` : (
                          item.result_details?.predicted_class || 'Multi-Output'
                        )}
                      </td>
                      <td className="py-3 px-4 text-right text-slate-300">
                        {item.actual_value !== null ? `${item.actual_value} ${item.unit || ''}` : (
                          item.result_details?.actual_class || '—'
                        )}
                      </td>
                      <td className="py-3 px-4 text-right">
                        {item.residual !== null ? (
                          <span className={Math.abs(item.residual) < 50 ? 'text-emerald-400' : 'text-amber-400'}>
                            {item.residual > 0 ? `+${item.residual}` : item.residual}
                          </span>
                        ) : '—'}
                      </td>
                      <td className="py-3 px-4 text-center">
                        <button
                          onClick={() => handleDelete(item.id)}
                          className="p-1 rounded hover:bg-rose-950 text-slate-400 hover:text-rose-400 transition"
                          title="Delete record"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
