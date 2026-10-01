import React, { useState, useEffect } from 'react';
import { 
  GitBranch, Info, AlertCircle, ArrowUpRight, ArrowDownRight, 
  Layers, Filter, HelpCircle 
} from 'lucide-react';
import api from '../api';

export default function ExplainabilityPage() {
  const [target, setTarget] = useState('peak_flood_level');
  const [topN, setTopN] = useState(15);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const targets = [
    { id: 'peak_flood_level', label: 'Peak Flood Level (m)' },
    { id: 'peak_discharge', label: 'Peak Discharge Q (cumec)' },
    { id: 'flood_volume', label: 'Flood Volume (cumec)' },
  ];

  useEffect(() => {
    async function loadShap() {
      try {
        setLoading(true);
        setError(null);
        const res = await api.getShapAnalytics(target, topN);
        setData(res);
      } catch (err) {
        setError(err.message || 'Failed to fetch SHAP analysis');
      } finally {
        setLoading(false);
      }
    }
    loadShap();
  }, [target, topN]);

  const maxShap = data?.top_features?.[0]?.mean_abs_shap || 1;

  return (
    <div className="space-y-6 pb-16">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center space-x-3">
          <GitBranch className="w-8 h-8 text-cyan-400" />
          <span>Explainable AI — SHAP Feature Attribution</span>
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Audited Shapley Additive Explanations (SHAP) decomposing model contributions across watershed morphometry and climate indices.
        </p>
      </div>

      {/* Prominent Scientific Disclaimer Banner */}
      <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/80 text-amber-200 text-xs sm:text-sm flex items-start space-x-3 shadow-lg shadow-amber-950/20">
        <Info className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <strong className="text-white font-bold block mb-1">
            "SHAP values indicate model contribution, not causal relationships."
          </strong>
          <span className="text-slate-300 text-xs leading-relaxed">
            High positive or negative SHAP attributions reflect statistical associations within the trained gradient boosting regression trees. 
            They quantify how much a feature shifts the prediction relative to the baseline expectation across the IndoFloods dataset, but do not imply direct physical causation without empirical hydraulic verification.
          </span>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl bg-slate-900 border border-slate-800 gap-4">
        {/* Target Buttons */}
        <div className="flex flex-wrap gap-2">
          {targets.map((t) => (
            <button
              key={t.id}
              onClick={() => setTarget(t.id)}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
                target === t.id
                  ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                  : 'bg-slate-950 text-slate-300 hover:text-white border border-slate-800'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Top N Filter */}
        <div className="flex items-center space-x-2 text-xs">
          <span className="text-slate-400">Display Top:</span>
          <select
            value={topN}
            onChange={(e) => setTopN(parseInt(e.target.value))}
            className="px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value={10}>10 Features</option>
            <option value={15}>15 Features</option>
            <option value={20}>20 Features</option>
            <option value={30}>30 Features</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center min-h-[40vh]">
          <div className="w-8 h-8 border-4 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin"></div>
        </div>
      ) : error ? (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-xs">
          {error}
        </div>
      ) : (
        <div className="space-y-6">
          {/* Top Features Visual Bar Chart */}
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="font-bold text-white text-base">
                  Top {data?.top_features?.length} Drivers of {data?.target}
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Mean absolute SHAP value (|SHAP|) representing relative magnitude of influence across the test split.
                </p>
              </div>
              <span className="text-xs px-2 py-1 rounded bg-slate-800 text-slate-300 font-mono">
                {data?.model_name}
              </span>
            </div>

            {/* Horizontal Bar Chart representation */}
            <div className="space-y-3">
              {data?.top_features?.map((feat, idx) => {
                const widthPct = Math.max(4, Math.round((feat.mean_abs_shap / maxShap) * 100));
                const isPos = feat.direction === 'Positive';

                return (
                  <div key={feat.feature} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center space-x-2 font-medium">
                        <span className="w-5 text-slate-500 font-mono text-[11px]">#{idx + 1}</span>
                        <span className="text-slate-200">{feat.display_name}</span>
                        <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-400">
                          {feat.category}
                        </span>
                      </div>

                      <div className="flex items-center space-x-3 font-mono">
                        {feat.direction && (
                          <span className={`text-[11px] flex items-center space-x-0.5 ${
                            isPos ? 'text-emerald-400' : 'text-rose-400'
                          }`}>
                            {isPos ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                            <span>{feat.direction}</span>
                          </span>
                        )}
                        <span className="text-cyan-300 font-bold">
                          {feat.mean_abs_shap.toFixed(2)}
                        </span>
                      </div>
                    </div>

                    {/* Bar track */}
                    <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden flex border border-slate-800/80">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          isPos ? 'bg-gradient-to-r from-cyan-500 to-teal-400' : 'bg-gradient-to-r from-blue-500 to-indigo-500'
                        }`}
                        style={{ width: `${widthPct}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Detailed Attribution Table */}
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800">
            <h3 className="font-bold text-white text-base mb-1">
              Audited Feature Attribution & Directional Breakdown
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Breakdown of directional influence showing percentage of test events with positive vs negative contributions.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 text-xs uppercase font-medium">
                    <th className="py-2.5 px-3">Rank</th>
                    <th className="py-2.5 px-3">Predictor Feature</th>
                    <th className="py-2.5 px-3">Category</th>
                    <th className="py-2.5 px-3 text-right">Mean |SHAP|</th>
                    <th className="py-2.5 px-3 text-center">Net Direction</th>
                    <th className="py-2.5 px-3 text-right">Positive Impact %</th>
                    <th className="py-2.5 px-3 text-right">Negative Impact %</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {data?.top_features?.map((f, idx) => (
                    <tr key={f.feature} className="hover:bg-slate-800/30">
                      <td className="py-2.5 px-3 text-slate-500">#{idx + 1}</td>
                      <td className="py-2.5 px-3 font-sans font-medium text-white">{f.display_name}</td>
                      <td className="py-2.5 px-3 font-sans text-slate-400">{f.category}</td>
                      <td className="py-2.5 px-3 text-right font-bold text-cyan-300">{f.mean_abs_shap.toFixed(2)}</td>
                      <td className="py-2.5 px-3 text-center font-sans">
                        <span className={`px-2 py-0.5 rounded text-[11px] ${
                          f.direction === 'Positive'
                            ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                            : 'bg-rose-950 text-rose-300 border border-rose-800'
                        }`}>
                          {f.direction || 'Neutral'}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-right text-emerald-400">
                        {f.positive_contribution_pct !== null ? `${f.positive_contribution_pct}%` : '—'}
                      </td>
                      <td className="py-2.5 px-3 text-right text-rose-400">
                        {f.negative_contribution_pct !== null ? `${f.negative_contribution_pct}%` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
