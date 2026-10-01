import React from 'react';
import { Waves, ShieldCheck, Activity, Database, GitBranch, AlertTriangle } from 'lucide-react';

export default function Header({ activeTab, setActiveTab, health }) {
  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'prediction', label: 'Flood Prediction', icon: Waves },
    { id: 'classification', label: 'Flood Classification', icon: AlertTriangle },
    { id: 'explainability', label: 'Model Explainability (SHAP)', icon: GitBranch },
    { id: 'analytics', label: 'Hydrological Analytics', icon: Database },
    { id: 'history', label: 'Prediction History', icon: ShieldCheck },
  ];

  const isHealthy = health?.status === 'healthy';

  return (
    <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Title */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-500 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-white font-bold">
              <Waves className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg tracking-tight text-white">AI Water Intelligence</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-950 border border-cyan-700/50 text-cyan-300 font-mono">
                  v1.0 Research
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">Disaster Resilience & Hydrological Platform</p>
            </div>
          </div>

          {/* System Health / Audit Badges */}
          <div className="hidden lg:flex items-center space-x-3 text-xs">
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-emerald-950/60 border border-emerald-800/50 text-emerald-300">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span>Audit: 27 PASS / 0 FAIL</span>
            </div>
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700 text-slate-300 font-mono">
              <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
              <span>{isHealthy ? 'Models Operational' : 'Connecting...'}</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex space-x-1 overflow-x-auto py-2 border-t border-slate-800/60 text-sm no-scrollbar">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg font-medium whitespace-nowrap transition-all duration-150 ${
                  isActive
                    ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-sm shadow-cyan-500/10'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
