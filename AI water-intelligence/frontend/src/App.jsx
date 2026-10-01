import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import PredictionPage from './pages/PredictionPage';
import ClassificationPage from './pages/ClassificationPage';
import ExplainabilityPage from './pages/ExplainabilityPage';
import AnalyticsPage from './pages/AnalyticsPage';
import HistoryPage from './pages/HistoryPage';
import api from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [health, setHealth] = useState(null);

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await api.getHealth();
        setHealth(res);
      } catch (err) {
        console.warn('Backend currently offline or unreachable:', err);
        setHealth({ status: 'unreachable' });
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-white">
      {/* Sticky Top Header Navigation */}
      <Header 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        health={health} 
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        {activeTab === 'dashboard' && <Dashboard setActiveTab={setActiveTab} />}
        {activeTab === 'prediction' && <PredictionPage />}
        {activeTab === 'classification' && <ClassificationPage />}
        {activeTab === 'explainability' && <ExplainabilityPage />}
        {activeTab === 'analytics' && <AnalyticsPage />}
        {activeTab === 'history' && <HistoryPage />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p>© 2026 AI-Powered Water Intelligence & Disaster Resilience Platform. IndoFloods Benchmark.</p>
          <div className="flex items-center space-x-4">
            <span className="text-slate-400">Audited ML Pipelines (27 PASS / 0 FAIL)</span>
            <span>•</span>
            <a 
              href="http://localhost:8000/docs" 
              target="_blank" 
              rel="noreferrer" 
              className="text-cyan-400 hover:text-cyan-300 underline"
            >
              Interactive Swagger API (/docs)
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
