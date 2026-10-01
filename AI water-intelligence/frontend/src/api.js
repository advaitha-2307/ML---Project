const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {})
  };

  try {
    const res = await fetch(url, { ...options, headers });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Request failed with status ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    console.error(`API Error on ${endpoint}:`, error);
    throw error;
  }
}

export const api = {
  getHealth: () => request('/health'),
  getFeaturesMetadata: () => request('/metadata/features'),
  getAnalyticsSummary: () => request('/analytics/summary'),
  getShapAnalytics: (target = 'peak_flood_level', topN = 15) => request(`/analytics/shap/${target}?top_n=${topN}`),
  getResiduals: (target = 'peak_flood_level') => request(`/analytics/residuals/${target}`),
  getErrorAnalysis: (target = 'peak_flood_level') => request(`/analytics/error-analysis/${target}`),
  
  getEvents: (params = {}) => {
    const query = new URLSearchParams();
    if (params.page) query.set('page', params.page);
    if (params.pageSize) query.set('page_size', params.pageSize);
    if (params.search) query.set('search', params.search);
    if (params.gaugeId) query.set('gauge_id', params.gaugeId);
    if (params.floodType) query.set('flood_type', params.floodType);
    if (params.basin) query.set('basin', params.basin);
    return request(`/events?${query.toString()}`);
  },
  
  getEventDetail: (eventId) => request(`/events/${encodeURIComponent(eventId)}`),
  getGauges: () => request('/gauges'),
  getGaugeDetail: (gaugeId) => request(`/gauges/${encodeURIComponent(gaugeId)}`),

  predictFloodLevel: (payload) => request('/predict/flood-level', { method: 'POST', body: JSON.stringify(payload) }),
  predictDischarge: (payload) => request('/predict/discharge', { method: 'POST', body: JSON.stringify(payload) }),
  predictVolume: (payload) => request('/predict/volume', { method: 'POST', body: JSON.stringify(payload) }),
  predictFloodType: (payload) => request('/predict/flood-type', { method: 'POST', body: JSON.stringify(payload) }),
  predictAll: (payload) => request('/predict/all', { method: 'POST', body: JSON.stringify(payload) }),
  explain: (payload) => request('/explain', { method: 'POST', body: JSON.stringify(payload) }),

  getHistory: (limit = 50) => request(`/history?limit=${limit}`),
  deleteHistoryItem: (id) => request(`/history/${id}`, { method: 'DELETE' }),
};

export default api;
