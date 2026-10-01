import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, ShieldCheck, Play, Info, CheckCircle2, 
  HelpCircle, RefreshCw, BarChart2, Check
} from 'lucide-react';
import api from '../api';

export default function ClassificationPage() {
  const [events, setEvents] = useState([]);
  const [selectedEventId, setSelectedEventId] = useState('');
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadInitial() {
      try {
        const res = await api.getEvents({ pageSize: 20 });
        setEvents(res.items || []);
        if (res.items?.length > 0) {
          setSelectedEventId(res.items[0].EventID);
          loadEventDetail(res.items[0].EventID);
        }
      } catch (err) {
        console.error('Failed to load events:', err);
      }
    }
    loadInitial();
  }, []);

  async function loadEventDetail(eventId) {
    setSelectedEventId(eventId);
    setResult(null);
    try {
      const detail = await api.getEventDetail(eventId);
      setSelectedEvent(detail);
    } catch (err) {
      console.error(err);
    }
  }

  async function handleClassify() {
    if (!selectedEventId) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.predictFloodType({ event_id: selectedEventId });
      setResult(res);
    } catch (err) {
      setError(err.message || 'Classification request failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6 pb-16">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center space-x-3">
          <AlertTriangle className="w-8 h-8 text-amber-400" />
          <span>Flood Type Severity Classification</span>
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Binary classification of flood events into Standard Flood vs Severe Flood using the trained Gradient Boosting Classifier.
        </p>
      </div>

      {/* Model Benchmark Card */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400 uppercase font-medium">Model Architecture</span>
          <div className="text-lg font-bold text-cyan-400 mt-1">Gradient Boosting</div>
          <p className="text-[11px] text-slate-400 mt-1">Pipeline with ColumnTransformer</p>
        </div>
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400 uppercase font-medium">Overall Accuracy</span>
          <div className="text-2xl font-bold text-white font-mono mt-1">70.05%</div>
          <p className="text-[11px] text-slate-400 mt-1">IndoFloods Test Split (788 events)</p>
        </div>
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400 uppercase font-medium">Severe Flood Precision</span>
          <div className="text-2xl font-bold text-emerald-400 font-mono mt-1">62.73%</div>
          <p className="text-[11px] text-slate-400 mt-1">Positive class discrimination</p>
        </div>
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400 uppercase font-medium">ROC-AUC Metric</span>
          <div className="text-2xl font-bold text-indigo-400 font-mono mt-1">0.6782</div>
          <p className="text-[11px] text-slate-400 mt-1">Classification ranking ability</p>
        </div>
      </div>

      {/* Interactive Classification Workbench */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Selector & Run Column */}
        <div className="lg:col-span-1 p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <h3 className="font-bold text-white text-sm">Select Event for Severity Assessment</h3>
          <p className="text-xs text-slate-400">
            Choose an event from the integrated dataset to evaluate retrospective classification confidence.
          </p>

          <div>
            <label className="text-xs text-slate-300 font-medium block mb-1">Select EventID:</label>
            <select
              value={selectedEventId}
              onChange={(e) => loadEventDetail(e.target.value)}
              className="w-full px-3 py-2 text-xs rounded-xl bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
            >
              {events.map((ev) => (
                <option key={ev.EventID} value={ev.EventID}>
                  {ev.EventID} — {ev.Station || 'Station'} ({ev.Flood_Type || 'Flood'})
                </option>
              ))}
            </select>
          </div>

          {selectedEvent && (
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800/80 space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-400">Actual Category:</span>
                <span className={`font-bold font-mono ${selectedEvent['Flood Type'] === 'Severe Flood' ? 'text-amber-400' : 'text-cyan-400'}`}>
                  {selectedEvent['Flood Type'] || 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Station / Gauge:</span>
                <span className="text-slate-200">{selectedEvent.Station || 'Station'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">River Basin:</span>
                <span className="text-slate-200">{selectedEvent.Basin || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Start Date:</span>
                <span className="text-slate-200 font-mono">{selectedEvent['Start Date'] || 'N/A'}</span>
              </div>
            </div>
          )}

          <button
            onClick={handleClassify}
            disabled={loading || !selectedEventId}
            className="w-full py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-slate-950 font-bold text-xs shadow-lg shadow-amber-500/20 transition flex items-center justify-center space-x-2 cursor-pointer"
          >
            {loading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-slate-950/20 border-t-slate-950 rounded-full animate-spin"></div>
                <span>Classifying Event...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Run Severity Classification</span>
              </>
            )}
          </button>
        </div>

        {/* Result & Confidence Output */}
        <div className="lg:col-span-2 p-6 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
              <h3 className="font-bold text-white text-base flex items-center space-x-2">
                <BarChart2 className="w-4 h-4 text-cyan-400" />
                <span>Classification Assessment Output</span>
              </h3>
              <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                {result ? result.status : 'Awaiting input'}
              </span>
            </div>

            {error && (
              <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800 text-rose-300 text-xs">
                {error}
              </div>
            )}

            {result ? (
              <div className="space-y-6">
                {/* Prediction Result Pill */}
                <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
                  <div>
                    <span className="text-xs text-slate-400 uppercase font-medium">Model Classification Decision</span>
                    <div className="text-3xl font-extrabold mt-1 flex items-center space-x-3">
                      <span className={result.predicted_class === 'Severe Flood' ? 'text-amber-400' : 'text-cyan-400'}>
                        {result.predicted_class}
                      </span>
                      {result.actual_class && (
                        <span className={`text-xs px-2 py-1 rounded font-normal border ${
                          result.predicted_class === result.actual_class
                            ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                            : 'bg-rose-950 text-rose-300 border-rose-800'
                        }`}>
                          {result.predicted_class === result.actual_class ? '✓ Matches Ground Truth' : 'Mismatch'}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="text-right">
                    <span className="text-xs text-slate-400 uppercase font-medium">Confidence</span>
                    <div className="text-2xl font-bold text-white font-mono mt-1">
                      {result.confidence ? `${Math.round(result.confidence * 100)}%` : 'N/A'}
                    </div>
                  </div>
                </div>

                {/* Probability Distribution Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-cyan-400">
                      Standard Flood Probability: {result.probability_flood ? `${Math.round(result.probability_flood * 100)}%` : 'N/A'}
                    </span>
                    <span className="text-amber-400">
                      Severe Flood Probability: {result.probability_severe_flood ? `${Math.round(result.probability_severe_flood * 100)}%` : 'N/A'}
                    </span>
                  </div>

                  <div className="w-full bg-slate-950 h-3 rounded-full overflow-hidden flex border border-slate-800">
                    <div 
                      className="bg-cyan-500 h-full transition-all duration-500" 
                      style={{ width: `${(result.probability_flood || 0.5) * 100}%` }}
                    ></div>
                    <div 
                      className="bg-amber-500 h-full transition-all duration-500" 
                      style={{ width: `${(result.probability_severe_flood || 0.5) * 100}%` }}
                    ></div>
                  </div>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-300 space-y-1">
                  <p><strong>Actual Historical Class:</strong> {result.actual_class || 'N/A'}</p>
                  <p><strong>Evaluation Mode:</strong> {result.mode}</p>
                  <p className="text-slate-400 text-[11px] mt-2 italic">{result.disclaimer}</p>
                </div>
              </div>
            ) : (
              <div className="py-16 text-center text-slate-500 text-xs">
                Select an event and click "Run Severity Classification" to view predicted class and probabilities.
              </div>
            )}
          </div>

          {/* Research Architecture Note */}
          <div className="mt-6 pt-4 border-t border-slate-800/80 text-xs text-slate-400 space-y-1.5">
            <div className="flex items-center space-x-1.5 text-slate-300 font-semibold">
              <Info className="w-3.5 h-3.5 text-cyan-400" />
              <span>Feature Architecture & Integrity Separation:</span>
            </div>
            <p className="text-[11px] leading-relaxed">
              The classification pipeline uses 118 features (including antecedent warning levels, entries, and reliability), 
              whereas the three controlled regression models use exactly 113 features (where quality and threshold indicators were deliberately removed 
              in accordance with Final Integrity Audit Check #9 to eliminate target leakage).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
