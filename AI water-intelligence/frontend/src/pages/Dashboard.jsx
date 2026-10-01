import React, { useState, useEffect } from 'react';
import { 
  Waves, AlertTriangle, Gauge, CheckCircle2, 
  ArrowRight, ShieldCheck, BarChart3, TrendingUp, Cpu
} from 'lucide-react';
import api from '../api';

export default function Dashboard({ setActiveTab }) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const data = await api.getAnalyticsSummary();
        setSummary(data);
      } catch (err) {
        setError(err.message || 'Failed to load analytics summary');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center space-y-3">
          <div className="w-10 h-10 border-4 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin"></div>
          <p className="text-slate-400 text-sm">Loading research benchmark summary...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300">
          <p className="font-semibold">Unable to connect to backend API:</p>
          <p className="text-sm mt-1">{error}</p>
          <p className="text-xs mt-2 text-rose-400">Ensure the FastAPI backend is running on http://localhost:8000</p>
        </div>
      </div>
    );
  }

  const floodCount = summary?.flood_count || 2919;
  const severeCount = summary?.severe_flood_count || 1629;
  const totalEvents = summary?.total_events || 4548;
  const floodPct = Math.round((floodCount / totalEvents) * 100);
  const severePct = Math.round((severeCount / totalEvents) * 100);

  return (
    <div className="space-y-8 pb-12">
      {/* Hero / Overview Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-cyan-950/60 border border-slate-800 p-6 sm:p-8">
        <div className="absolute right-0 top-0 -mt-10 -mr-10 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="max-w-3xl">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-800/60 text-cyan-300 text-xs font-medium mb-4">
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
            <span>Audited Hydrological Benchmark (27 / 27 PASS)</span>
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight text-white">
            AI-Powered Water Intelligence & Disaster Resilience Platform
          </h1>
          <p className="mt-3 text-slate-300 text-sm sm:text-base leading-relaxed">
            A research-validated hydrological framework for retrospective event replay, scenario-based flood simulation, 
            and explainable machine learning across 155 river gauges from the IndoFloods benchmark.
          </p>

          <div className="mt-6 flex flex-wrap gap-3">
            <button
              onClick={() => setActiveTab('prediction')}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-sm transition shadow-lg shadow-cyan-500/20"
            >
              <span>Explore Flood Predictions</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => setActiveTab('explainability')}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-sm border border-slate-700 transition"
            >
              <span>View SHAP Attribution</span>
            </button>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Events */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800/80 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">Total Flood Events</span>
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400">
              <Waves className="w-5 h-5" />
            </div>
          </div>
          <p className="mt-3 text-3xl font-bold text-white tracking-tight">{totalEvents.toLocaleString()}</p>
          <p className="mt-1 text-xs text-slate-400">Integrated historical flood hydrographs</p>
        </div>

        {/* Flood vs Severe Flood */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800/80 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">Event Severity</span>
            <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400">
              <AlertTriangle className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3 flex items-baseline space-x-3">
            <span className="text-xl font-bold text-slate-200">{floodCount} <span className="text-xs font-normal text-slate-400">Flood</span></span>
            <span className="text-slate-600">|</span>
            <span className="text-xl font-bold text-amber-400">{severeCount} <span className="text-xs font-normal text-amber-500/80">Severe</span></span>
          </div>
          {/* Progress bar */}
          <div className="mt-2.5 w-full bg-slate-800 h-2 rounded-full overflow-hidden flex">
            <div className="bg-cyan-500 h-full" style={{ width: `${floodPct}%` }} title={`Flood: ${floodPct}%`}></div>
            <div className="bg-amber-500 h-full" style={{ width: `${severePct}%` }} title={`Severe Flood: ${severePct}%`}></div>
          </div>
          <div className="mt-1.5 flex justify-between text-[11px] text-slate-400">
            <span>Flood ({floodPct}%)</span>
            <span>Severe ({severePct}%)</span>
          </div>
        </div>

        {/* Gauges */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800/80 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">Available Gauges</span>
            <div className="p-2 rounded-lg bg-teal-500/10 text-teal-400">
              <Gauge className="w-5 h-5" />
            </div>
          </div>
          <p className="mt-3 text-3xl font-bold text-white tracking-tight">{summary?.total_gauges || 155}</p>
          <p className="mt-1 text-xs text-slate-400">Pan-India river monitoring stations</p>
        </div>

        {/* Model Capabilities */}
        <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800/80 backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">ML Capabilities</span>
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
              <Cpu className="w-5 h-5" />
            </div>
          </div>
          <p className="mt-3 text-3xl font-bold text-emerald-400 tracking-tight">3 Reg + 1 Clf</p>
          <p className="mt-1 text-xs text-slate-400">Peak Level, Discharge, Volume, Type</p>
        </div>
      </div>

      {/* Benchmark Performance Matrix */}
      <div className="rounded-2xl bg-slate-900/90 border border-slate-800 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-cyan-400" />
              <span>Controlled Regression Benchmark Results</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Exact metrics computed from audited test splits (106 numerical + 7 categorical features, target leakage excluded).
            </p>
          </div>
          <div className="inline-flex items-center space-x-2 text-xs px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span>Integrity: {summary?.integrity_audit_status}</span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-xs font-medium uppercase tracking-wider">
                <th className="py-3 px-4">Target Variable</th>
                <th className="py-3 px-4">Selected Final Model</th>
                <th className="py-3 px-4 text-right">R² Score</th>
                <th className="py-3 px-4 text-right">MAE</th>
                <th className="py-3 px-4 text-right">RMSE</th>
                <th className="py-3 px-4 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-4 font-sans font-medium text-white flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                  <span>Peak Flood Level (m)</span>
                </td>
                <td className="py-3 px-4 text-cyan-300">Gradient Boosting</td>
                <td className="py-3 px-4 text-right font-bold text-emerald-400">0.8848</td>
                <td className="py-3 px-4 text-right text-slate-300">31.10 m</td>
                <td className="py-3 px-4 text-right text-slate-300">61.90 m</td>
                <td className="py-3 px-4 text-center">
                  <span className="px-2 py-0.5 rounded-full text-xs bg-emerald-950 text-emerald-400 border border-emerald-800/60">
                    Deployed
                  </span>
                </td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-4 font-sans font-medium text-white flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-blue-400"></span>
                  <span>Peak Discharge Q (cumec)</span>
                </td>
                <td className="py-3 px-4 text-cyan-300">Gradient Boosting</td>
                <td className="py-3 px-4 text-right font-bold text-cyan-400">0.4176</td>
                <td className="py-3 px-4 text-right text-slate-300">1,413.55 cumec</td>
                <td className="py-3 px-4 text-right text-slate-300">2,531.34 cumec</td>
                <td className="py-3 px-4 text-center">
                  <span className="px-2 py-0.5 rounded-full text-xs bg-emerald-950 text-emerald-400 border border-emerald-800/60">
                    Deployed
                  </span>
                </td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-3 px-4 font-sans font-medium text-white flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-indigo-400"></span>
                  <span>Flood Volume (cumec)</span>
                </td>
                <td className="py-3 px-4 text-cyan-300">Gradient Boosting</td>
                <td className="py-3 px-4 text-right font-bold text-cyan-400">0.2808</td>
                <td className="py-3 px-4 text-right text-slate-300">5,299.01 cumec</td>
                <td className="py-3 px-4 text-right text-slate-300">9,096.71 cumec</td>
                <td className="py-3 px-4 text-center">
                  <span className="px-2 py-0.5 rounded-full text-xs bg-emerald-950 text-emerald-400 border border-emerald-800/60">
                    Deployed
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Suite Details Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Classification Model Summary */}
        <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-white text-base flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <span>Flood Type Classification Benchmark</span>
              </h3>
              <span className="text-xs px-2 py-0.5 rounded bg-amber-950/70 border border-amber-800 text-amber-300">
                Binary Classifier
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-2">
              Trained on IndoFloods classification splits to discriminate standard floods from severe inundation.
            </p>

            <div className="mt-4 grid grid-cols-3 gap-2 text-center">
              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/50">
                <div className="text-lg font-bold text-cyan-400 font-mono">70.05%</div>
                <div className="text-[11px] text-slate-400 mt-0.5">Accuracy</div>
              </div>
              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/50">
                <div className="text-lg font-bold text-emerald-400 font-mono">62.73%</div>
                <div className="text-[11px] text-slate-400 mt-0.5">Severe Precision</div>
              </div>
              <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/50">
                <div className="text-lg font-bold text-indigo-400 font-mono">0.6782</div>
                <div className="text-[11px] text-slate-400 mt-0.5">ROC-AUC</div>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs">
            <span className="text-slate-400">Saved Model: GradientBoostingClassifier</span>
            <button 
              onClick={() => setActiveTab('classification')}
              className="text-cyan-400 hover:text-cyan-300 font-medium inline-flex items-center space-x-1"
            >
              <span>Test Classification</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Explainability & Diagnostics Overview */}
        <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-white text-base flex items-center space-x-2">
                <BarChart3 className="w-4 h-4 text-cyan-400" />
                <span>Explainable AI & Diagnostics</span>
              </h3>
              <span className="text-xs px-2 py-0.5 rounded bg-cyan-950/70 border border-cyan-800 text-cyan-300">
                SHAP + Residuals
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-2">
              Full feature attribution and error decomposition across catchment morphometry, precipitation, and regimes.
            </p>

            <ul className="mt-4 space-y-2 text-xs text-slate-300">
              <li className="flex items-start space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5"></span>
                <span><strong className="text-white">SHAP Analysis:</strong> Feature contributions across 113 parameters without causal claims.</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5"></span>
                <span><strong className="text-white">Residual Distributions:</strong> Actual vs predicted scatter and error histograms for model diagnostics.</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5"></span>
                <span><strong className="text-white">Gauge Regimes:</strong> Error behavior stratified across high relief vs low relief basins.</span>
              </li>
            </ul>
          </div>

          <div className="mt-5 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs">
            <span className="text-slate-400">Reports: 27 PASS audited</span>
            <button 
              onClick={() => setActiveTab('analytics')}
              className="text-cyan-400 hover:text-cyan-300 font-medium inline-flex items-center space-x-1"
            >
              <span>Explore Analytics</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
