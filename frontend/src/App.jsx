import React, { useState, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { Terminal, Sparkles } from "lucide-react";
import logo from "./assets/microstack-logo.png";
import "./App.css";

// Use relative URLs - Vite proxy handles routing to backend
const API_BASE_URL = "";

function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [logs, setLogs] = useState("");
  const [showLogs, setShowLogs] = useState(false);
  const [assetContents, setAssetContents] = useState({});

  // Fetch text content for non-image assets like .xyz
  useEffect(() => {
    if (results?.all_images) {
      results.all_images.forEach(async (asset) => {
        if (asset.endsWith(".xyz") && !assetContents[asset]) {
          try {
            const response = await axios.get(getImageUrl(asset));
            setAssetContents((prev) => ({ ...prev, [asset]: response.data }));
          } catch (err) {
            console.error("Failed to fetch asset content", err);
          }
        }
      });
    }
  }, [results]);

  // Auto-expand textarea height
  useEffect(() => {
    const textarea = document.getElementById("query-input");
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [query]);

  useEffect(() => {
    let interval;
    if (loading) {
      const fetchLogs = async () => {
        try {
          const response = await axios.get(`${API_BASE_URL}/api/logs?lines=50`);
          setLogs(response.data.logs);
        } catch (err) {
          console.error("Failed to fetch logs", err);
        }
      };
      interval = setInterval(fetchLogs, 1000);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResults(null);
    setError(null);
    setLogs("µ-Stack engine started...");

    try {
      const response = await axios.post(`${API_BASE_URL}/api/query`, {
        query,
        session_id: sessionId,
      });

      const logResponse = await axios.get(`${API_BASE_URL}/api/logs?lines=200`);
      setLogs(logResponse.data.logs);

      setResults(response.data.results);
      setSessionId(response.data.session_id);
    } catch (err) {
      setError(err.response?.data?.detail || "An error occurred.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getImageUrl = (path) => {
    if (!path) return null;
    const parts = path.split(/[\\/]/);
    const outputIndex = parts.indexOf("output");
    if (outputIndex !== -1) {
      const relativePath = parts.slice(outputIndex + 1).join("/");
      return `${API_BASE_URL}/output/${relativePath}`;
    }
    return null;
  };

  return (
    <div className="app-container">
      <header>
        <img src={logo} alt="µ-Stack Logo" />
        <p>AI Materials Scientist · Zoom In To Atomic Scale</p>
      </header>

      <section className="query-section">
        <form onSubmit={handleSubmit}>
          <textarea
            id="query-input"
            className="query-input"
            placeholder="What should we simulate today?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
            rows={1}
          />
          <div className="button-group">
            {results && (
              <button
                type="button"
                className="secondary-button"
                onClick={() => setShowLogs(!showLogs)}
              >
                <Terminal size={16} />
                {showLogs ? "Hide Logs" : "View Logs"}
              </button>
            )}
            <button type="submit" disabled={loading || !query.trim()}>
              {loading ? (
                <div className="spinner"></div>
              ) : (
                <>
                  <Sparkles size={18} />
                  Run Simulation
                </>
              )}
            </button>
          </div>
        </form>
      </section>

      {loading && (
        <div className="log-viewer">
          <div className="log-content">{logs}</div>
        </div>
      )}

      {showLogs && results && (
        <div className="log-viewer">
          <div className="log-content">{logs}</div>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      {results && (
        <main>
          {results.ai_summary_md && (
            <section className="summary-card">
              <h2>AI Scientific Summary</h2>
              <div className="markdown-body">
                <ReactMarkdown>{results.ai_summary_md}</ReactMarkdown>
              </div>
            </section>
          )}

          {results.all_images && results.all_images.length > 0 && (
            <section className="image-gallery">
              <h2>Simulation Results</h2>
              <div className="results-grid">
                {results.all_images
                  .filter((img) => !img.toLowerCase().includes("unrelaxed"))
                  .map((img, idx) => (
                    <div key={idx} className="result-card">
                      <h3>{img.split(/[\\/]/).pop()}</h3>
                      <div className="result-image-container">
                        {img.endsWith(".xyz") ? (
                          <pre className="xyz-viewer">
                            {assetContents[img] || "Loading structure data..."}
                          </pre>
                        ) : (
                          <img
                            src={getImageUrl(img)}
                            alt="Simulation result"
                            className="result-image"
                          />
                        )}
                      </div>
                    </div>
                  ))}
              </div>
            </section>
          )}

          {results.report_md && (
            <section className="summary-card">
              <div className="markdown-body">
                <ReactMarkdown>{results.report_md}</ReactMarkdown>
              </div>
            </section>
          )}
        </main>
      )}
    </div>
  );
}

export default App;
