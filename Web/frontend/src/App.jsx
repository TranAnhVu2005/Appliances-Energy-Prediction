import { useState } from 'react'
import './index.css'

const categories = {
  temperature: {
    label: "Nhiệt độ (°C)",
    features: ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T_out']
  },
  humidity: {
    label: "Độ ẩm (%)",
    features: ['RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_6', 'RH_7', 'RH_8', 'RH_9', 'RH_out']
  },
  environment: {
    label: "Thời tiết & Môi trường",
    features: ['Press_mm_hg', 'Windspeed', 'Visibility', 'Tdewpoint', 'lights']
  },
  time: {
    label: "Thời gian",
    features: ['Hour', 'DayOfWeek']
  }
}

// Labels for individual features to make UI nicer
const featureLabels = {
  T1: "Bếp (T1)", RH_1: "Độ ẩm Bếp (RH_1)",
  T2: "Phòng khách (T2)", RH_2: "Độ ẩm P.Khách (RH_2)",
  T3: "Phòng giặt (T3)", RH_3: "Độ ẩm P.Giặt (RH_3)",
  T4: "Phòng làm việc (T4)", RH_4: "Độ ẩm P.Làm việc (RH_4)",
  T5: "Phòng tắm (T5)", RH_5: "Độ ẩm P.Tắm (RH_5)",
  T6: "Bên ngoài Bắc (T6)", RH_6: "Độ ẩm Ngoài Bắc (RH_6)",
  T7: "Phòng ủi (T7)", RH_7: "Độ ẩm P.Ủi (RH_7)",
  T8: "Phòng Teen (T8)", RH_8: "Độ ẩm P.Teen (RH_8)",
  T9: "Phòng ngủ cha mẹ (T9)", RH_9: "Độ ẩm P.Ngủ (RH_9)",
  T_out: "Nhiệt độ ngoài trời (T_out)", RH_out: "Độ ẩm ngoài trời (RH_out)",
  Press_mm_hg: "Áp suất (mm_hg)", Windspeed: "Tốc độ gió (m/s)",
  Visibility: "Tầm nhìn (km)", Tdewpoint: "Điểm sương (Tdewpoint)",
  lights: "Năng lượng đèn (Wh)",
  Hour: "Giờ (0-23)", DayOfWeek: "Ngày trong tuần (0=T2, 6=CN)"
}

// Default values based loosely on dataset averages to avoid typing 27 numbers
const initialFeatures = {
  lights: 10, T1: 21.6, RH_1: 40.0, T2: 20.5, RH_2: 40.0,
  T3: 22.0, RH_3: 38.0, T4: 20.0, RH_4: 40.0, T5: 19.5, RH_5: 50.0,
  T6: 10.0, RH_6: 50.0, T7: 20.0, RH_7: 35.0, T8: 22.0, RH_8: 45.0,
  T9: 20.0, RH_9: 40.0, T_out: 10.0, Press_mm_hg: 750.0, RH_out: 80.0,
  Windspeed: 5.0, Visibility: 40.0, Tdewpoint: 5.0, Hour: 12, DayOfWeek: 2
}

function App() {
  const [activeTab, setActiveTab] = useState('temperature');
  const [features, setFeatures] = useState(initialFeatures);
  const [model, setModel] = useState('Random_Forest');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (feature, value) => {
    setFeatures(prev => ({
      ...prev,
      [feature]: value
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://127.0.0.1:8080/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: model,
          features: features
        })
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed');
      }

      setResult(data.prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>Dự đoán Năng lượng Thiết bị</h1>
        <p className="subtitle">Ứng dụng Machine Learning phân tích và dự báo tiêu thụ điện (Wh)</p>
      </header>

      <main className="glass-panel">
        <div className="tabs">
          {Object.entries(categories).map(([key, category]) => (
            <button 
              key={key}
              className={`tab-btn ${activeTab === key ? 'active' : ''}`}
              onClick={() => setActiveTab(key)}
            >
              {category.label}
            </button>
          ))}
        </div>

        <div className="input-grid">
          {categories[activeTab].features.map(feature => (
            <div className="input-group" key={feature}>
              <label>{featureLabels[feature] || feature}</label>
              <input 
                type="number" 
                step="any"
                value={features[feature]}
                onChange={(e) => handleInputChange(feature, e.target.value)}
              />
            </div>
          ))}
        </div>

        <div className="actions">
          <div className="model-selector">
            <label>Mô hình dự đoán:</label>
            <select value={model} onChange={e => setModel(e.target.value)}>
              <option value="KNN">K-Nearest Neighbors (KNN)</option>
              <option value="Decision_Tree">Decision Tree</option>
              <option value="Random_Forest">Random Forest (Đề xuất)</option>
              <option value="Linear_Regression">Linear Regression</option>
            </select>
          </div>

          <button 
            className="btn-predict" 
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? <span className="spinner"></span> : 'BẮT ĐẦU DỰ ĐOÁN'}
          </button>

          {error && (
            <div className="error-message">
              ⚠️ Lỗi: {error}
            </div>
          )}

          {result !== null && (
            <div className="result-card glass-panel">
              <h2>Kết quả dự đoán (Mô hình {model.replace('_', ' ')})</h2>
              <div className="result-value">
                {result.toFixed(2)} <span className="result-unit">Wh</span>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
