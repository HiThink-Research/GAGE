const resolveConfigValue = (key, envKey, fallback) => {
    if (typeof window !== 'undefined') {
        const windowConfig = window.__GAGE_CONFIG__ || {};
        if (windowConfig[key]) {
            return windowConfig[key];
        }
    }
    if (typeof process !== 'undefined' && process.env && process.env[envKey]) {
        return process.env[envKey];
    }
    return fallback;
};

const apiUrl = resolveConfigValue('apiUrl', 'REACT_APP_GAGE_API_URL', 'http://127.0.0.1:8000');
const actionUrl = resolveConfigValue('actionUrl', 'REACT_APP_GAGE_ACTION_URL', apiUrl);
const douzeroDemoUrl = resolveConfigValue(
    'douzeroDemoUrl',
    'REACT_APP_DOUZERO_DEMO_URL',
    'http://127.0.0.1:5000',
);

export { apiUrl, actionUrl, douzeroDemoUrl };
