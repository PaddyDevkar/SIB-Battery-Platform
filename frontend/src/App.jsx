import { useState } from "react";
import HDF5Explorer from "./components/HDF5Explorer.jsx";

function App() {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("analyzer");

  const handleAnalyze = async () => {
    if (!file) {
      alert("Select file first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setData(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const json = await res.json();
      setData(json);
    } catch (err) {
      console.error(err);
      alert("Prediction failed");
    }

    setLoading(false);
  };

  return (
    <div
      style={{
        height: "100vh",
        width: "100vw",
        display: "flex",
        flexDirection: "column",
        background: "#0b1220",
        color: "white",
        overflow: "hidden",
        fontFamily: "Inter, sans-serif",
      }}
    >
      {/* ================= HEADER ================= */}
      <div
        style={{
          padding: "20px 40px",
          borderBottom: "1px solid #1f2937",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <h1 style={{ margin: 0, fontSize: 34 }}>
          🔋 SIB Battery Intelligence Dashboard
        </h1>

        <div style={{ marginTop: 15, display: "flex", gap: 20 }}>
          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
          />

          <button
            onClick={() => setActiveTab("analyzer")}
            style={{
              padding: "8px 16px",
              background:
                activeTab === "analyzer" ? "#2563eb" : "#1f2937",
              border: "none",
              borderRadius: 6,
              color: "white",
              cursor: "pointer",
            }}
          >
            Analyzer
          </button>

          <button
            onClick={() => setActiveTab("explorer")}
            style={{
              padding: "8px 16px",
              background:
                activeTab === "explorer" ? "#2563eb" : "#1f2937",
              border: "none",
              borderRadius: 6,
              color: "white",
              cursor: "pointer",
            }}
          >
            HDF5 Explorer
          </button>
        </div>
      </div>

      {/* ================= MAIN CONTENT ================= */}
      <div
        style={{
          flex: 1,
          overflow: "hidden",
          display: "flex",
        }}
      >
        {/* ANALYZER VIEW */}
        {activeTab === "analyzer" && (
          <div
            style={{
              padding: 40,
              width: "100%",
              overflowY: "auto",
            }}
          >
            <button
              onClick={handleAnalyze}
              style={{
                padding: "10px 20px",
                background: "#16a34a",
                border: "none",
                borderRadius: 8,
                color: "white",
                fontSize: 16,
                cursor: "pointer",
              }}
            >
              Analyze Battery
            </button>

            {loading && (
              <p style={{ marginTop: 20 }}>Analyzing...</p>
            )}

            {data && (
              <div style={{ marginTop: 40 }}>
                <h2>Battery Health Overview</h2>

                <div style={{ marginTop: 20, lineHeight: 1.8 }}>
                  <p>
                    <strong>Health Score:</strong>{" "}
                    {data.battery_analysis.health_score}
                  </p>
                  <p>
                    <strong>Risk Level:</strong>{" "}
                    {data.battery_analysis.risk_level}
                  </p>
                  <p>
                    <strong>SOH (Energy):</strong>{" "}
                    {data.battery_analysis.soh_energy_percent}%
                  </p>
                  <p>
                    <strong>SOH (Power):</strong>{" "}
                    {data.battery_analysis.soh_power_percent}%
                  </p>
                  <p>
                    <strong>Remaining Useful Life:</strong>{" "}
                    {data.battery_analysis
                      .remaining_useful_life_cycles}{" "}
                    cycles
                  </p>
                  <p>
                    <strong>Degradation Mode:</strong>{" "}
                    {data.battery_analysis.degradation_mode}
                  </p>
                  <p>
                    <strong>Failure Probability:</strong>{" "}
                    {data.battery_analysis
                      .failure_probability_percent}%
                  </p>
                  <p>
                    <strong>Confidence Score:</strong>{" "}
                    {data.battery_analysis.confidence_score}%
                  </p>
                </div>

                <div style={{ marginTop: 30 }}>
                  <h3>AI Interpretation</h3>
                  <p style={{ lineHeight: 1.6 }}>
                    {data.human_interpretation}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* HDF5 EXPLORER VIEW */}
        {activeTab === "explorer" && (
          <div style={{ flex: 1 }}>
            <HDF5Explorer file={file} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
