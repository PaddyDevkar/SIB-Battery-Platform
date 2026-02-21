import { useState, useMemo } from "react";
import Plot from "react-plotly.js";

function HDF5Explorer({ file }) {
  const [tree, setTree] = useState(null);
  const [selectedPath, setSelectedPath] = useState(null);
  const [datasetData, setDatasetData] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  // Controls
  const [plotMode, setPlotMode] = useState("line");
  const [logScale, setLogScale] = useState(false);
  const [stride, setStride] = useState(1);
  const [smoothing, setSmoothing] = useState(1);
  const [showFFT, setShowFFT] = useState(false);

  // ================================
  // LOAD STRUCTURE
  // ================================
  const loadStructure = async () => {
    if (!file) return alert("Upload file first");

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/hdf5/structure", {
      method: "POST",
      body: formData,
    });

    const json = await res.json();
    setTree(json.tree);
  };

  // ================================
  // LOAD DATASET
  // ================================
  const loadDataset = async (path) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("dataset_path", path);
    formData.append("stride", stride);

    setLoading(true);

    const res = await fetch("http://127.0.0.1:8000/hdf5/dataset", {
      method: "POST",
      body: formData,
    });

    const json = await res.json();

    setDatasetData(json.data);
    setStats(json.stats);
    setSelectedPath(path);
    setLoading(false);
  };

  // ================================
  // FAST SMOOTHING (optimized)
  // ================================
  const smoothData = (data, windowSize) => {
    if (windowSize <= 1) return data;

    const result = [];
    for (let i = 0; i < data.length; i++) {
      let sum = 0;
      let count = 0;
      for (let j = -windowSize; j <= windowSize; j++) {
        const idx = i + j;
        if (idx >= 0 && idx < data.length) {
          sum += data[idx];
          count++;
        }
      }
      result.push(sum / count);
    }
    return result;
  };

  // ================================
  // LIGHTWEIGHT FFT (fast enough)
  // ================================
  const computeFFT = (data) => {
    const N = Math.min(data.length, 2048); // limit size for speed
    const output = [];

    for (let k = 0; k < N / 2; k++) {
      let re = 0;
      let im = 0;
      for (let n = 0; n < N; n++) {
        const angle = (2 * Math.PI * k * n) / N;
        re += data[n] * Math.cos(angle);
        im -= data[n] * Math.sin(angle);
      }
      output.push(Math.sqrt(re * re + im * im));
    }

    return output;
  };

  // ================================
  // PROCESS DATA (memoized = smoother)
  // ================================
  const processedData = useMemo(() => {
    if (!datasetData.length) return [];

    const smoothed = smoothData(datasetData, smoothing);
    return showFFT ? computeFFT(smoothed) : smoothed;
  }, [datasetData, smoothing, showFFT]);

  // ================================
  // TREE RENDER
  // ================================
  const renderTree = (nodes) =>
    nodes.map((node) => (
      <div key={node.path} style={{ marginLeft: 15 }}>
        {node.type === "group" ? (
          <>
            📂 <strong>{node.name}</strong>
            {node.children && renderTree(node.children)}
          </>
        ) : (
          <div
            style={{
              cursor: "pointer",
              color: "#38bdf8",
              marginTop: 4,
            }}
            onClick={() => loadDataset(node.path)}
          >
            📄 {node.name}
          </div>
        )}
      </div>
    ));

  return (
    <div
      style={{
        display: "flex",
        height: "100%",
        width: "100%",
        overflow: "hidden",
      }}
    >
      {/* LEFT SIDEBAR */}
      <div
        style={{
          width: "300px",
          background: "#111827",
          padding: 20,
          overflowY: "auto",
          borderRight: "1px solid #1f2937",
        }}
      >
        <button
          onClick={loadStructure}
          style={{
            background: "#2563eb",
            border: "none",
            padding: "8px 14px",
            borderRadius: 6,
            color: "white",
            cursor: "pointer",
            marginBottom: 15,
          }}
        >
          Load Structure
        </button>

        {tree && renderTree(tree)}
      </div>

      {/* CENTER PLOT */}
      <div
        style={{
          flex: 1,
          background: "#0b1220",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {loading && <p>Loading...</p>}

        {processedData.length > 0 && (
          <Plot
            data={[
              {
                x: processedData.map((_, i) => i),
                y: processedData,
                type: "scattergl", // GPU acceleration
                mode: plotMode === "line" ? "lines" : "markers",
                marker: { size: 3 },
                line: { width: 1 },
              },
            ]}
            layout={{
              title: selectedPath,
              autosize: true,
              margin: { t: 40, l: 50, r: 30, b: 40 },
              yaxis: { type: logScale ? "log" : "linear" },
              paper_bgcolor: "#0b1220",
              plot_bgcolor: "#0b1220",
              font: { color: "white" },
            }}
            config={{
              responsive: true,
              displaylogo: false,
              scrollZoom: true,
            }}
            style={{ width: "95%", height: "90%" }}
          />
        )}
      </div>

      {/* RIGHT CONTROLS */}
      <div
        style={{
          width: "320px",
          background: "#111827",
          padding: 20,
          overflowY: "auto",
          borderLeft: "1px solid #1f2937",
        }}
      >
        <h3>Controls</h3>

        <label>Plot Mode</label>
        <select
          value={plotMode}
          onChange={(e) => setPlotMode(e.target.value)}
        >
          <option value="line">Line</option>
          <option value="scatter">Scatter</option>
        </select>

        <br /><br />

        <label>
          <input
            type="checkbox"
            checked={logScale}
            onChange={() => setLogScale(!logScale)}
          />
          Log Scale
        </label>

        <br /><br />

        <label>Stride (Downsample)</label>
        <input
          type="range"
          min="1"
          max="20"
          value={stride}
          onChange={(e) => setStride(parseInt(e.target.value))}
        />

        <br /><br />

        <label>Smoothing</label>
        <input
          type="range"
          min="1"
          max="20"
          value={smoothing}
          onChange={(e) => setSmoothing(parseInt(e.target.value))}
        />

        <br /><br />

        <label>
          <input
            type="checkbox"
            checked={showFFT}
            onChange={() => setShowFFT(!showFFT)}
          />
          FFT View
        </label>

        <br /><br />

        {stats && (
          <>
            <h4>Statistics</h4>
            <p>Min: {stats.min}</p>
            <p>Max: {stats.max}</p>
            <p>Mean: {stats.mean}</p>
            <p>Std: {stats.std}</p>
            <p>Size: {stats.size}</p>
          </>
        )}
      </div>
    </div>
  );
}

export default HDF5Explorer;
