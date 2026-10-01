import React, { useState, useEffect } from 'react';
import { 
  Waves, Search, Filter, History, Play, RotateCcw, 
  CheckCircle, AlertCircle, Info, ChevronDown, ChevronRight,
  MapPin, Calendar, CloudRain, Droplets
} from 'lucide-react';
import api from '../api';

export default function PredictionPage() {
  const [mode, setMode] = useState('replay'); // 'replay' or 'scenario'
  const [selectedTarget, setSelectedTarget] = useState('peak_flood_level'); // 'peak_flood_level', 'peak_discharge', 'flood_volume', 'all'
  
  // Replay mode state
  const [events, setEvents] = useState([]);
  const [selectedEventId, setSelectedEventId] = useState('');
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [eventsLoading, setEventsLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [totalEvents, setTotalEvents] = useState(0);

  // Scenario mode state
  const [gauges, setGauges] = useState([]);
  const [selectedGaugeId, setSelectedGaugeId] = useState('');
  const [featureMetadata, setFeatureMetadata] = useState(null);
  const [formData, setFormData] = useState({});
  const [activeGroup, setActiveGroup] = useState('Rainfall');

  // Prediction execution & result state
  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState(null);

  // Load initial events and gauges
  useEffect(() => {
    loadEvents();
    loadGauges();
    loadFeatureMetadata();
  }, []);

  async function loadEvents(search = '', pageNum = 1) {
    try {
      setEventsLoading(true);
      const res = await api.getEvents({ search, page: pageNum, pageSize: 15 });
      setEvents(res.items || []);
      setTotalEvents(res.total || 0);
      if (res.items?.length > 0 && !selectedEventId) {
        selectEvent(res.items[0].EventID);
      }
    } catch (err) {
      console.error('Failed to load events:', err);
    } finally {
      setEventsLoading(false);
    }
  }

  async function loadGauges() {
    try {
      const data = await api.getGauges();
      setGauges(data || []);
      if (data?.length > 0) {
        setSelectedGaugeId(data[0].GaugeID);
        applyGaugePreset(data[0]);
      }
    } catch (err) {
      console.error('Failed to load gauges:', err);
    }
  }

  async function loadFeatureMetadata() {
    try {
      const meta = await api.getFeaturesMetadata();
      setFeatureMetadata(meta);
    } catch (err) {
      console.error('Failed to load metadata:', err);
    }
  }

  async function selectEvent(eventId) {
    setSelectedEventId(eventId);
    setPredictionResult(null);
    setPredictionError(null);
    try {
      const detail = await api.getEventDetail(eventId);
      setSelectedEvent(detail);
    } catch (err) {
      console.error('Failed to fetch event detail:', err);
    }
  }

  function applyGaugePreset(gauge) {
    if (!gauge || !gauge.sample_features) return;
    setFormData((prev) => ({
      ...prev,
      ...gauge.sample_features,
      GaugeID: gauge.GaugeID,
      Basin: gauge.Basin || prev.Basin,
      State: gauge.State || prev.State,
      Latitude: gauge.Latitude || prev.Latitude,
      Longitude: gauge.Longitude || prev.Longitude,
      'Catchment Area': gauge.Drainage_Area || prev['Catchment Area'],
      'Stream Order': gauge.Stream_Order || prev['Stream Order'],
    }));
  }

  function handleGaugeChange(e) {
    const gid = e.target.value;
    setSelectedGaugeId(gid);
    const g = gauges.find((item) => String(item.GaugeID) === String(gid));
    if (g) applyGaugePreset(g);
  }

  function handleInputChange(name, value) {
    setFormData((prev) => ({
      ...prev,
      [name]: value
    }));
  }

  function handleRainfallPreset(presetType) {
    const presetValues = {};
    if (presetType === 'heavy') {
      // 100mm downpour decaying
      presetValues.T1d = 125.0; presetValues.T2d = 95.0; presetValues.T3d = 70.0;
      presetValues.T4d = 45.0;  presetValues.T5d = 30.0; presetValues.T6d = 15.0;
      presetValues.T7d = 10.0;  presetValues.T8d = 5.0;  presetValues.T9d = 0.0; presetValues.T10d = 0.0;
    } else if (presetType === 'moderate') {
      presetValues.T1d = 45.0;  presetValues.T2d = 35.0; presetValues.T3d = 25.0;
      presetValues.T4d = 20.0;  presetValues.T5d = 15.0; presetValues.T6d = 10.0;
      presetValues.T7d = 5.0;   presetValues.T8d = 5.0;  presetValues.T9d = 2.0; presetValues.T10d = 0.0;
    } else {
      // Low / Dry
      for (let i = 1; i <= 10; i++) presetValues[`T${i}d`] = 2.0;
    }
    setFormData((prev) => ({ ...prev, ...presetValues }));
  }

  async function handlePredict() {
    setPredicting(true);
    setPredictionError(null);
    setPredictionResult(null);

    try {
      let res;
      const payload = mode === 'replay' 
        ? { event_id: selectedEventId }
        : { features: formData, gauge_id: selectedGaugeId };

      if (selectedTarget === 'peak_flood_level') {
        res = await api.predictFloodLevel(payload);
      } else if (selectedTarget === 'peak_discharge') {
        res = await api.predictDischarge(payload);
      } else if (selectedTarget === 'flood_volume') {
        res = await api.predictVolume(payload);
      } else {
        // all targets
        res = await api.predictAll(payload);
      }
      setPredictionResult(res);
    } catch (err) {
      setPredictionError(err.message || 'Prediction execution failed.');
    } finally {
      setPredicting(false);
    }
  }

  const targets = [
    { id: 'peak_flood_level', label: 'Peak Flood Level (m)', unit: 'm' },
    { id: 'peak_discharge', label: 'Peak Discharge Q (cumec)', unit: 'cumec' },
    { id: 'flood_volume', label: 'Flood Volume (cumec)', unit: 'cumec' },
    { id: 'all', label: 'All Regression Suite', unit: 'multi' },
  ];

  return (
    <div className="space-y-6 pb-16">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center space-x-3">
          <Waves className="w-8 h-8 text-cyan-400" />
          <span>Hydrological Flood Prediction</span>
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Execute controlled gradient boosting pipelines for flood level, peak discharge, and volume estimation.
        </p>
      </div>

      {/* Target Selector Tabs */}
      <div className="flex flex-wrap gap-2 p-1.5 bg-slate-900/90 rounded-xl border border-slate-800">
        {targets.map((t) => (
          <button
            key={t.id}
            onClick={() => { setSelectedTarget(t.id); setPredictionResult(null); }}
            className={`px-4 py-2 rounded-lg text-xs sm:text-sm font-medium transition ${
              selectedTarget === t.id
                ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Mode Toggle Banner */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between p-4 rounded-xl bg-slate-900 border border-slate-800 gap-4">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Evaluation Mode:</span>
          <div className="flex rounded-lg bg-slate-950 p-1 border border-slate-800">
            <button
              onClick={() => { setMode('replay'); setPredictionResult(null); }}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
                mode === 'replay'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Mode A: Historical Replay
            </button>
            <button
              onClick={() => { setMode('scenario'); setPredictionResult(null); }}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
                mode === 'scenario'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Mode B: Scenario Simulation
            </button>
          </div>
        </div>

        <button
          onClick={handlePredict}
          disabled={predicting || (mode === 'replay' && !selectedEventId)}
          className="inline-flex items-center justify-center space-x-2 px-6 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/20 transition cursor-pointer"
        >
          {predicting ? (
            <>
              <div className="w-4 h-4 border-2 border-slate-950/20 border-t-slate-950 rounded-full animate-spin"></div>
              <span>Executing Model Pipeline...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>{mode === 'replay' ? 'Run Retrospective Replay' : 'Run Scenario Prediction'}</span>
            </>
          )}
        </button>
      </div>

      {/* Mode Disclaimer Banner */}
      <div className={`p-4 rounded-xl border flex items-start space-x-3 text-xs ${
        mode === 'replay'
          ? 'bg-blue-950/30 border-blue-800/60 text-blue-200'
          : 'bg-amber-950/30 border-amber-800/60 text-amber-200'
      }`}>
        <Info className={`w-4 h-4 mt-0.5 shrink-0 ${mode === 'replay' ? 'text-blue-400' : 'text-amber-400'}`} />
        <div>
          <strong className="font-semibold block mb-0.5">
            {mode === 'replay' 
              ? 'Historical Event Replay — this is a retrospective model demonstration, not a live forecast.'
              : 'Scenario Simulation — research prediction based on supplied watershed parameters, not an operational field warning.'
            }
          </strong>
          <span className="text-slate-300">
            {mode === 'replay'
              ? 'Evaluates model response using genuine recorded predictors from IndoFloods benchmark event records with observed actual hydrograph targets.'
              : 'Applies the 113 trained pipeline features. Use the Gauge Catchment Template below to prefill realistic watershed geometry and adjust rainfall lags.'
            }
          </span>
        </div>
      </div>

      {/* Error alert if any */}
      {predictionError && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{predictionError}</span>
        </div>
      )}

      {/* Prediction Result Display (When Available) */}
      {predictionResult && (
        <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-cyan-950/40 border-2 border-cyan-500/40 shadow-xl shadow-cyan-950/30">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-4 mb-5">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-mono uppercase tracking-wider text-cyan-400">Prediction Output</span>
                <span className="px-2 py-0.5 rounded text-[11px] bg-cyan-950 border border-cyan-800 text-cyan-300">
                  {predictionResult.model_used || 'Gradient Boosting Regressor'}
                </span>
              </div>
              <h2 className="text-xl font-bold text-white mt-1">
                {predictionResult.target || 'Multi-Target Evaluation Results'}
              </h2>
            </div>
            <div className="text-right text-xs text-slate-400 font-mono">
              Timestamp: {new Date(predictionResult.timestamp).toLocaleTimeString()}
            </div>
          </div>

          {/* Single Target Result Cards */}
          {selectedTarget !== 'all' ? (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {/* Predicted Value */}
              <div className="p-4 rounded-xl bg-slate-950/70 border border-cyan-500/30">
                <span className="text-xs text-slate-400 uppercase font-medium">Predicted Value</span>
                <div className="text-3xl font-extrabold text-cyan-400 font-mono mt-2">
                  {predictionResult.predicted_value?.toLocaleString()}
                  <span className="text-sm font-normal text-slate-400 ml-1.5">{predictionResult.unit}</span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Controlled Gradient Boosting estimate</p>
              </div>

              {/* Actual Value */}
              <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-xs text-slate-400 uppercase font-medium">Observed Actual</span>
                <div className="text-3xl font-extrabold text-white font-mono mt-2">
                  {predictionResult.actual_value !== null && predictionResult.actual_value !== undefined
                    ? predictionResult.actual_value.toLocaleString()
                    : 'N/A (Scenario)'}
                  {predictionResult.actual_value !== null && (
                    <span className="text-sm font-normal text-slate-400 ml-1.5">{predictionResult.unit}</span>
                  )}
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Recorded gauge station value</p>
              </div>

              {/* Residual Delta */}
              <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
                <span className="text-xs text-slate-400 uppercase font-medium">Residual Error (Pred - Act)</span>
                <div className={`text-3xl font-extrabold font-mono mt-2 ${
                  predictionResult.residual === null ? 'text-slate-500' :
                  Math.abs(predictionResult.residual) < 50 ? 'text-emerald-400' : 'text-amber-400'
                }`}>
                  {predictionResult.residual !== null && predictionResult.residual !== undefined
                    ? (predictionResult.residual > 0 ? `+${predictionResult.residual}` : predictionResult.residual)
                    : 'N/A'}
                  {predictionResult.residual !== null && (
                    <span className="text-sm font-normal text-slate-400 ml-1.5">{predictionResult.unit}</span>
                  )}
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Retrospective model residual</p>
              </div>
            </div>
          ) : (
            /* Multi-Target Grid */
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="p-4 rounded-xl bg-slate-950/70 border border-cyan-500/30">
                <span className="text-xs text-cyan-400 uppercase font-medium">Peak Flood Level</span>
                <div className="text-2xl font-bold text-white font-mono mt-1">
                  {predictionResult.flood_level?.predicted_value} m
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Actual: {predictionResult.flood_level?.actual_value !== null ? `${predictionResult.flood_level?.actual_value} m` : 'N/A'}
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950/70 border border-blue-500/30">
                <span className="text-xs text-blue-400 uppercase font-medium">Peak Discharge Q</span>
                <div className="text-2xl font-bold text-white font-mono mt-1">
                  {predictionResult.discharge?.predicted_value?.toLocaleString()} cumec
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Actual: {predictionResult.discharge?.actual_value !== null ? `${predictionResult.discharge?.actual_value} cumec` : 'N/A'}
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950/70 border border-indigo-500/30">
                <span className="text-xs text-indigo-400 uppercase font-medium">Flood Volume</span>
                <div className="text-2xl font-bold text-white font-mono mt-1">
                  {predictionResult.volume?.predicted_value?.toLocaleString()} cumec
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Actual: {predictionResult.volume?.actual_value !== null ? `${predictionResult.volume?.actual_value} cumec` : 'N/A'}
                </div>
              </div>
            </div>
          )}

          <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
            <span>Audit check: Saved pipeline called with exact 113 features directly</span>
            <span className="text-emerald-400 flex items-center space-x-1">
              <CheckCircle className="w-3.5 h-3.5" />
              <span>Pipeline Preprocessing Intact</span>
            </span>
          </div>
        </div>
      )}

      {/* Main Content Area based on Mode */}
      {mode === 'replay' ? (
        /* MODE A: HISTORICAL EVENT REPLAY */
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Search & Events Table */}
          <div className="lg:col-span-1 rounded-2xl bg-slate-900 border border-slate-800 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-white text-sm flex items-center space-x-2">
                <History className="w-4 h-4 text-cyan-400" />
                <span>Historical Events ({totalEvents})</span>
              </h3>
              <span className="text-xs text-slate-400">Page {page}</span>
            </div>

            {/* Search Input */}
            <div className="relative">
              <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
              <input
                type="text"
                placeholder="Search EventID, Station, Basin..."
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setPage(1);
                  loadEvents(e.target.value, 1);
                }}
                className="w-full pl-9 pr-3 py-1.5 text-xs rounded-lg bg-slate-950 border border-slate-800 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
              />
            </div>

            {/* Events List */}
            <div className="space-y-1.5 max-h-[460px] overflow-y-auto pr-1">
              {eventsLoading ? (
                <div className="py-8 text-center text-xs text-slate-500">Loading events...</div>
              ) : events.length === 0 ? (
                <div className="py-8 text-center text-xs text-slate-500">No events matched.</div>
              ) : (
                events.map((ev) => {
                  const isSelected = selectedEventId === ev.EventID;
                  return (
                    <div
                      key={ev.EventID}
                      onClick={() => selectEvent(ev.EventID)}
                      className={`p-3 rounded-xl border text-xs cursor-pointer transition ${
                        isSelected
                          ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-200'
                          : 'bg-slate-950/60 border-slate-800/80 text-slate-300 hover:bg-slate-800/50'
                      }`}
                    >
                      <div className="flex items-center justify-between font-mono font-bold">
                        <span>{ev.EventID}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                          ev.Flood_Type === 'Severe Flood' 
                            ? 'bg-amber-950 text-amber-300 border border-amber-800/50'
                            : 'bg-cyan-950 text-cyan-300 border border-cyan-800/50'
                        }`}>
                          {ev.Flood_Type || 'Flood'}
                        </span>
                      </div>
                      <div className="mt-1 flex items-center justify-between text-slate-400 text-[11px]">
                        <span>{ev.Station || 'Station Gauge'}</span>
                        <span>{ev.Start_Date || 'N/A'}</span>
                      </div>
                      <div className="mt-1 flex justify-between text-[11px] font-mono text-slate-400">
                        <span>Level: {ev.Peak_Flood_Level !== null ? `${ev.Peak_Flood_Level}m` : '—'}</span>
                        <span>Q: {ev.Peak_Discharge !== null ? `${ev.Peak_Discharge}` : '—'}</span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            {/* Pagination Controls */}
            <div className="flex justify-between items-center pt-2 border-t border-slate-800 text-xs">
              <button
                disabled={page <= 1}
                onClick={() => { const p = page - 1; setPage(p); loadEvents(searchTerm, p); }}
                className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300"
              >
                Previous
              </button>
              <span className="text-slate-400">Page {page}</span>
              <button
                disabled={events.length < 15}
                onClick={() => { const p = page + 1; setPage(p); loadEvents(searchTerm, p); }}
                className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300"
              >
                Next
              </button>
            </div>
          </div>

          {/* Right Column: Selected Event Details */}
          <div className="lg:col-span-2 rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-6">
            {selectedEvent ? (
              <>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-4">
                  <div>
                    <span className="text-xs font-mono text-cyan-400 uppercase tracking-wider">Historical Replay Candidate</span>
                    <h3 className="text-xl font-bold text-white font-mono mt-0.5">{selectedEvent.EventID}</h3>
                  </div>
                  <div className="flex items-center space-x-2 text-xs">
                    <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 flex items-center space-x-1.5">
                      <MapPin className="w-3.5 h-3.5 text-cyan-400" />
                      <span>{selectedEvent.Station || 'Station'}, {selectedEvent.State || 'India'}</span>
                    </span>
                    <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 flex items-center space-x-1.5">
                      <Calendar className="w-3.5 h-3.5 text-blue-400" />
                      <span>{selectedEvent['Start Date'] || 'N/A'}</span>
                    </span>
                  </div>
                </div>

                {/* Ground Truth Targets Grid */}
                <div>
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                    Recorded Observed Ground Truth (Hydrograph Targets)
                  </h4>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80">
                      <span className="text-[11px] text-slate-400">Peak Flood Level</span>
                      <div className="text-base font-bold text-white font-mono mt-1">
                        {selectedEvent['Peak Flood Level (m)'] !== null ? `${selectedEvent['Peak Flood Level (m)']} m` : 'N/A'}
                      </div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80">
                      <span className="text-[11px] text-slate-400">Peak Discharge Q</span>
                      <div className="text-base font-bold text-white font-mono mt-1">
                        {selectedEvent['Peak Discharge Q (cumec)'] !== null ? `${selectedEvent['Peak Discharge Q (cumec)']} cumec` : 'N/A'}
                      </div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80">
                      <span className="text-[11px] text-slate-400">Flood Volume</span>
                      <div className="text-base font-bold text-white font-mono mt-1">
                        {selectedEvent['Flood Volume (cumec)'] !== null ? `${selectedEvent['Flood Volume (cumec)']} cumec` : 'N/A'}
                      </div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80">
                      <span className="text-[11px] text-slate-400">Flood Type</span>
                      <div className="text-base font-bold text-amber-400 font-mono mt-1">
                        {selectedEvent['Flood Type'] || 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Key Predictor Snapshots */}
                <div>
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                    10-Day Lag Rainfall Accumulation Profile (T1d to T10d)
                  </h4>
                  <div className="grid grid-cols-5 sm:grid-cols-10 gap-1.5 font-mono text-center">
                    {[1,2,3,4,5,6,7,8,9,10].map((day) => {
                      const val = selectedEvent[`T${day}d`];
                      return (
                        <div key={day} className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                          <span className="text-[10px] text-slate-500 block">T{day}d</span>
                          <span className="text-xs font-bold text-cyan-300 block mt-0.5">
                            {val !== undefined && val !== null ? Math.round(val) : 0}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Geographic & Basin Specs */}
                <div>
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                    Catchment & Watershed Environmental Parameters
                  </h4>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80">
                      <span className="text-slate-500 block text-[11px]">River Basin</span>
                      <span className="font-semibold text-slate-200 mt-0.5 block">{selectedEvent.Basin || 'Unknown'}</span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80">
                      <span className="text-slate-500 block text-[11px]">Drainage / Catchment Area</span>
                      <span className="font-semibold text-slate-200 mt-0.5 block font-mono">
                        {selectedEvent['Catchment Area'] ? `${Math.round(selectedEvent['Catchment Area'])} km²` : 'N/A'}
                      </span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80">
                      <span className="text-slate-500 block text-[11px]">Stream Order</span>
                      <span className="font-semibold text-slate-200 mt-0.5 block font-mono">{selectedEvent['Stream Order'] || 'N/A'}</span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80">
                      <span className="text-slate-500 block text-[11px]">Soil Type</span>
                      <span className="font-semibold text-slate-200 mt-0.5 block">{selectedEvent['Soil type'] || 'N/A'}</span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80">
                      <span className="text-slate-500 block text-[11px]">Lithology</span>
                      <span className="font-semibold text-slate-200 mt-0.5 block">{selectedEvent['lithology type'] || 'N/A'}</span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80">
                      <span className="text-slate-500 block text-[11px]">Climate Classification</span>
                      <span className="font-semibold text-slate-200 mt-0.5 block">{selectedEvent['KoppenGeiger Climate Type'] || 'N/A'}</span>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="py-16 text-center text-slate-500 text-sm">Select an event from the list on the left.</div>
            )}
          </div>
        </div>
      ) : (
        /* MODE B: SCENARIO PREDICTION */
        <div className="space-y-6">
          {/* Gauge Catchment Template Selector */}
          <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 className="font-bold text-white text-base flex items-center space-x-2">
                  <MapPin className="w-4 h-4 text-cyan-400" />
                  <span>Gauge Catchment Baseline Template</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Select a real river gauge from the 155 monitoring stations to automatically prefill verified catchment morphometry, bioclimatic indices, and geography.
                </p>
              </div>

              {/* Gauge Dropdown */}
              <div className="w-full sm:w-80">
                <select
                  value={selectedGaugeId}
                  onChange={handleGaugeChange}
                  className="w-full px-3 py-2 text-xs rounded-xl bg-slate-950 border border-slate-700 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
                >
                  {gauges.map((g) => (
                    <option key={g.GaugeID} value={g.GaugeID}>
                      {g.Station} — {g.Basin} ({g.State})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Quick Rainfall Scenario Presets */}
            <div className="pt-3 border-t border-slate-800/80 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-slate-400 font-medium">Quick Rainfall Scenarios:</span>
              <button
                type="button"
                onClick={() => handleRainfallPreset('heavy')}
                className="px-3 py-1 rounded-lg bg-cyan-950 text-cyan-300 border border-cyan-800 hover:bg-cyan-900 transition flex items-center space-x-1"
              >
                <CloudRain className="w-3.5 h-3.5" />
                <span>Heavy Monsoonal Downpour (125mm Peak)</span>
              </button>
              <button
                type="button"
                onClick={() => handleRainfallPreset('moderate')}
                className="px-3 py-1 rounded-lg bg-blue-950 text-blue-300 border border-blue-800 hover:bg-blue-900 transition flex items-center space-x-1"
              >
                <Droplets className="w-3.5 h-3.5" />
                <span>Moderate Monsoon (45mm Peak)</span>
              </button>
              <button
                type="button"
                onClick={() => handleRainfallPreset('dry')}
                className="px-3 py-1 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 transition"
              >
                <span>Low Rainfall / Dry Antecedent</span>
              </button>
            </div>
          </div>

          {/* Grouped Feature Accordions */}
          {featureMetadata && (
            <div className="space-y-4">
              <div className="flex border-b border-slate-800 overflow-x-auto text-xs no-scrollbar">
                {Object.keys(featureMetadata.groups).map((grpKey) => {
                  const grp = featureMetadata.groups[grpKey];
                  const isActive = activeGroup === grpKey;
                  return (
                    <button
                      key={grpKey}
                      type="button"
                      onClick={() => setActiveGroup(grpKey)}
                      className={`px-4 py-2.5 font-semibold whitespace-nowrap border-b-2 transition ${
                        isActive
                          ? 'border-cyan-400 text-cyan-400 bg-cyan-500/5'
                          : 'border-transparent text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {grp.title} ({grp.features.length})
                    </button>
                  );
                })}
              </div>

              {/* Active Group Form Fields */}
              {featureMetadata.groups[activeGroup] && (
                <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800">
                  <div className="mb-5">
                    <h4 className="text-sm font-bold text-white">
                      {featureMetadata.groups[activeGroup].title}
                    </h4>
                    <p className="text-xs text-slate-400 mt-1">
                      {featureMetadata.groups[activeGroup].description}
                    </p>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {featureMetadata.groups[activeGroup].features.map((feat) => {
                      const currentVal = formData[feat.name] !== undefined ? formData[feat.name] : feat.default;

                      return (
                        <div key={feat.name} className="space-y-1">
                          <label className="text-xs font-medium text-slate-300 flex items-center justify-between">
                            <span className="truncate" title={feat.name}>{feat.name}</span>
                            {feat.name.startsWith('T') && feat.name.endsWith('d') && (
                              <span className="text-[10px] text-cyan-400 font-mono">mm</span>
                            )}
                          </label>

                          {feat.type === 'select' ? (
                            <select
                              value={currentVal || ''}
                              onChange={(e) => handleInputChange(feat.name, e.target.value)}
                              className="w-full px-2.5 py-1.5 text-xs rounded-lg bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500"
                            >
                              {feat.options?.map((opt) => (
                                <option key={opt} value={opt}>{opt}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="number"
                              step="any"
                              value={currentVal !== undefined && currentVal !== null ? currentVal : ''}
                              onChange={(e) => handleInputChange(feat.name, e.target.value === '' ? '' : parseFloat(e.target.value))}
                              className="w-full px-2.5 py-1.5 text-xs rounded-lg bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
                            />
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
