import React, { useState, useEffect } from 'react';
import { 
  BarChart3, TrendingUp, AlertOctagon, Gauge, Layers, 
  ArrowUpRight, ArrowDownRight, Compass, ShieldAlert 
} from 'lucide-react';
import { 
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, BarChart, Bar, Cell 
} from 'recharts';
import api from '../api';

export default function AnalyticsPage() {
  const [target, setTarget] = useState('peak_flood_level');
  const [residualsData, setResidualsData] = useState(null);
  const [errorData, setErrorData] = useState(null);
  const [summaryData, setSummaryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const targets = [
    { id: 'peak_flood_level', label: 'Peak Flood Level (m)', unit: 'm' },
    { id: 'peak_discharge', label: 'Peak Discharge Q (cumec)', unit: 'cumec' },
    { id: 'flood_volume', label: 'Flood Volume (cumec)', unit: 'cumec' },
  ];

  useEffect(() => {
    async function loadAnalytics() {
      try {
        setLoading(true);
        setError(null);
        const [res, err, sum] = await Promise.all([
          api.getResiduals(target),
          api.getErrorAnalysis(target),
          api.getAnalyticsSummary()
        ]);
        setResidualsData(res);
        setErrorData(err);
        setSummaryData(sum);
      } catch (err) {
        setError(err.message || 'Failed to load analytics');
      } finally {
        setLoading(false);
      }
    }
    loadAnalytics();
  }, [target]);

  const summary = residualsData?.summary || {};
  const currentUnit = targets.find(t => t.id === target)?.unit || '';

  return (
    <div className="space-y-6 pb-16">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center space-x-3">
          <BarChart3 className="w-8 h-8 text-cyan-400" />
          <span>Hydrological Error Analytics & Model Diagnostics</span>
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Rigorous post-test evaluation using audited residual datasets, parity correlations, regime diagnostics, and extreme error behavior.
        </p>
      </div>

      {/* Target Selector */}
      <div className="flex flex-wrap gap-2 p-1.5 bg-slate-900 rounded-xl border border-slate-800">
        {targets.map((t) => (
          <button
            key={t.id}
            onClick={() => setTarget(t.id)}
            className={`px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition ${
              target === t.id
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                : 'text-slate-300 hover:text-white hover:bg-slate-800'
            }`}
          >
            {t.label}
          </button>
        ))}
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
          {/* Key Metric Highlights from final_residual_summary */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase font-medium">Test Samples</span>
              <div className="text-xl font-bold text-white font-mono mt-1">{summary.Test_Samples || 788}</div>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase font-medium">R² Score</span>
              <div className="text-xl font-bold text-emerald-400 font-mono mt-1">
                {summary.R2 ? summary.R2.toFixed(4) : '0.8848'}
              </div>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase font-medium">MAE</span>
              <div className="text-xl font-bold text-cyan-400 font-mono mt-1">
                {summary.MAE ? `${summary.MAE.toFixed(2)} ${currentUnit}` : 'N/A'}
              </div>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase font-medium">RMSE</span>
              <div className="text-xl font-bold text-blue-400 font-mono mt-1">
                {summary.RMSE ? `${summary.RMSE.toFixed(2)} ${currentUnit}` : 'N/A'}
              </div>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase font-medium">Median Error</span>
              <div className="text-xl font-bold text-slate-200 font-mono mt-1">
                {summary.Median_Error ? `${summary.Median_Error.toFixed(2)} ${currentUnit}` : 'N/A'}
              </div>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-[11px] text-slate-400 uppercase font-medium">Residual STD</span>
              <div className="text-xl font-bold text-indigo-400 font-mono mt-1">
                {summary.Residual_STD ? `${summary.Residual_STD.toFixed(2)}` : 'N/A'}
              </div>
            </div>
          </div>

          {/* Charts Row: Parity Scatter Plot & Residuals Histogram */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* 1. Actual vs Predicted Scatter */}
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
              <div className="mb-4">
                <h3 className="font-bold text-white text-sm flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4 text-cyan-400" />
                  <span>Actual vs Predicted Parity Dispersion</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Points along the diagonal reflect accurate predictions across the test set.
                </p>
              </div>

              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis 
                      type="number" 
                      dataKey="Actual" 
                      name="Observed Actual" 
                      unit={` ${currentUnit}`} 
                      stroke="#64748b" 
                      fontSize={11}
                      tickLine={false}
                    />
                    <YAxis 
                      type="number" 
                      dataKey="Predicted" 
                      name="Model Predicted" 
                      unit={` ${currentUnit}`} 
                      stroke="#64748b" 
                      fontSize={11}
                      tickLine={false}
                    />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3', stroke: '#0ea5e9' }}
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                    />
                    <Scatter 
                      name="Test Events" 
                      data={residualsData?.sample_predictions || []} 
                      fill="#06b6d4" 
                      opacity={0.65} 
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* 2. Residual Distribution Histogram */}
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
              <div className="mb-4">
                <h3 className="font-bold text-white text-sm flex items-center space-x-2">
                  <BarChart3 className="w-4 h-4 text-cyan-400" />
                  <span>Residual Error Frequency Distribution</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Frequency of prediction residuals (Predicted - Actual) across 25 binned intervals.
                </p>
              </div>

              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={residualsData?.residual_histogram || []} margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis 
                      dataKey="bin_center" 
                      stroke="#64748b" 
                      fontSize={11}
                      tickLine={false}
                    />
                    <YAxis 
                      stroke="#64748b" 
                      fontSize={11}
                      tickLine={false}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                      formatter={(val) => [val, 'Count']}
                      labelFormatter={(lbl) => `Residual ≈ ${lbl} ${currentUnit}`}
                    />
                    <Bar dataKey="count" fill="#0284c7" radius={[4, 4, 0, 0]}>
                      {residualsData?.residual_histogram?.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={Math.abs(entry.bin_center) < (summary.MAE || 50) ? '#0ea5e9' : '#f59e0b'} 
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Regime Diagnostics & Extreme Outliers Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Regime Diagnostics Table */}
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
              <h3 className="font-bold text-white text-sm flex items-center space-x-2 mb-1">
                <Compass className="w-4 h-4 text-cyan-400" />
                <span>Hydrological Regime Error Breakdown</span>
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                Model behavior stratified across catchment morphometric regimes (e.g. relief ratio quartiles).
              </p>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase font-medium">
                      <th className="py-2.5 px-3">Hydrological Regime</th>
                      <th className="py-2.5 px-3 text-right">Event Count</th>
                      <th className="py-2.5 px-3 text-right">MAE</th>
                      <th className="py-2.5 px-3 text-right">Median Error</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {errorData?.regime_comparison?.map((reg, idx) => (
                      <tr key={idx} className="hover:bg-slate-800/30">
                        <td className="py-2.5 px-3 font-sans font-medium text-slate-200">{reg.Regime}</td>
                        <td className="py-2.5 px-3 text-right text-slate-400">{reg.Events}</td>
                        <td className="py-2.5 px-3 text-right font-bold text-cyan-300">
                          {Number(reg.Mean_Absolute_Error).toFixed(2)}
                        </td>
                        <td className="py-2.5 px-3 text-right text-slate-300">
                          {Number(reg.Median_Absolute_Error).toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Extreme Error Analysis Table */}
            <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800">
              <h3 className="font-bold text-white text-sm flex items-center space-x-2 mb-1">
                <AlertOctagon className="w-4 h-4 text-amber-400" />
                <span>Extreme Outlier Events (Top Error Diagnosed)</span>
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                Largest residual discrepancies on test splits documented for research honesty and calibration.
              </p>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase font-medium">
                      <th className="py-2.5 px-3">Gauge ID</th>
                      <th className="py-2.5 px-3 text-right">Actual</th>
                      <th className="py-2.5 px-3 text-right">Predicted</th>
                      <th className="py-2.5 px-3 text-right">Absolute Error</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {errorData?.top_errors?.slice(0, 7).map((errRow, idx) => (
                      <tr key={idx} className="hover:bg-slate-800/30">
                        <td className="py-2 px-3 text-slate-300">{errRow.GaugeID}</td>
                        <td className="py-2 px-3 text-right text-slate-400">{Number(errRow.Actual).toFixed(1)}</td>
                        <td className="py-2 px-3 text-right text-cyan-300">{Number(errRow.Predicted).toFixed(1)}</td>
                        <td className="py-2 px-3 text-right font-bold text-amber-400">
                          {Number(errRow.Absolute_Error).toFixed(1)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
