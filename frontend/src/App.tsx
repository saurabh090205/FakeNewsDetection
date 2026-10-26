import { useEffect, useMemo, useState } from "react";
import {
  Activity, AlertTriangle, ArrowRight, BarChart3, BookOpen, Brain, Check,
  CheckCircle2, ChevronRight, Clock, Cpu, Database, ExternalLink, FileSearch,
  FileText, HelpCircle, History, Info, Layers, Loader2, Menu, RefreshCw,
  RotateCcw, Search, Shield, ShieldAlert, Sparkles, Terminal, Trash2, X, Zap
} from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis
} from "recharts";
import { api } from "./services/api";
import type {
  AnalysisHistoryItem, DatasetStats, ExperimentCurves, Explanation, Metrics,
  ModelInfo, ModelName, Prediction, WordContribution
} from "./types/api";

type Page = "Analyze News" | "How It Works" | "Explainability" | "Model Registry" | "Performance" | "About";

const NAV_ITEMS: { name: Page; icon: typeof Search }[] = [
  { name: "Analyze News", icon: Search },
  { name: "How It Works", icon: Layers },
  { name: "Explainability", icon: Sparkles },
  { name: "Model Registry", icon: Brain },
  { name: "Performance", icon: BarChart3 },
  { name: "About", icon: Info },
];

const PRESET_EXAMPLES = [
  {
    tag: "real",
    label: "Reuters: Federal Reserve Policy",
    headline: "Federal Reserve Holds Interest Rates Steady Amid Stable Economic Indicators",
    article: "The central bank voted unanimously to maintain its benchmark federal funds rate during its scheduled policy meeting on Wednesday. Chair Jerome Powell indicated that while headline consumer price measures have moderated toward target thresholds, officials require further evidence of durable labor market equilibrium before commencing rate adjustments. Commercial markets experienced modest equity advances following the policy announcement, with treasury yields stabilizing across intermediate maturities. Financial institutions noted that credit spreads have remained well-ordered across consumer and corporate lending portfolios.",
  },
  {
    tag: "real",
    label: "Reuters: Clean Energy Transition",
    headline: "International Energy Agency Reports Accelerated Growth in Renewable Generation",
    article: "International energy delegates gathered in Geneva on Tuesday to review regional targets for solar and offshore wind capacity. According to the published summit communiqué, capital allocations for clean grid infrastructure increased by twelve percent over the prior fiscal calendar. Representatives from twenty-four nations agreed to establish unified standards for high-voltage interconnectors. Economic analysts noted that falling component prices and state incentives have sustained utility investments despite broader commercial headwinds across industrial manufacturing regions.",
  },
  {
    tag: "fake",
    label: "Sensationalist: Banned Miracle Cure",
    headline: "SHOCKING BOMBSHELL: Big Pharma Hiding Miracle 48-Hour Disease Cure",
    article: "Unbelievable leaked documents from anonymous pharmaceutical whistleblowers reveal that corrupt corporate executives are desperately hiding an ancient herbal tea formulation that permanently cures all chronic illnesses in forty-eight hours. Mainstream media networks are systematically censoring this incredible miracle discovery to protect their multi-billion dollar drug revenues. Brave independent healers are sharing this secret recipe across uncensored channels before corrupt health agencies permanently ban our accounts forever. Patriots must share this vital truth with everyone immediately.",
  },
  {
    tag: "fake",
    label: "Conspiracy: Secret Weather Machine",
    headline: "EXPOSED: Secret Subterranean Weather Machine Unleashed On Citizens",
    article: "Astonishing confidential military reports prove that shadowy globalist elites are actively manipulating severe weather anomalies using top-secret electromagnetic frequency emitters hidden inside abandoned missile silos. Whistleblower radar technicians confirm that catastrophic hurricanes and atmospheric heatwaves are being artificially engineered to destroy domestic food supplies and enforce totalitarian control. Corrupt corporate journalists refuse to investigate the horrifying evidence despite dozens of eyewitness testimonies from local electrical grid engineers.",
  },
];

const percent = (val?: number | null) => (val == null ? "—" : `${(val * 100).toFixed(1)}%`);
const numFmt = (val?: number | null) => (val == null ? "—" : new Intl.NumberFormat().format(val));

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>("Analyze News");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [metrics, setMetrics] = useState<Record<string, Metrics>>({});
  const [experiments, setExperiments] = useState<ExperimentCurves>({});
  const [datasetStats, setDatasetStats] = useState<DatasetStats | null>(null);
  const [minArticleWords, setMinArticleWords] = useState(50);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [apiError, setApiError] = useState("");

  // Analysis State
  const [headline, setHeadline] = useState(PRESET_EXAMPLES[0].headline);
  const [article, setArticle] = useState(PRESET_EXAMPLES[0].article);
  const [selectedModel, setSelectedModel] = useState<ModelName>("dense");
  const [activePresetIndex, setActivePresetIndex] = useState<number | null>(0);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [comparisonResults, setComparisonResults] = useState<Array<Prediction | { model: string; error: string; code: string }> | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState("");
  const [explainError, setExplainError] = useState("");
  const [textHighlightMode, setTextHighlightMode] = useState<"all" | "supports" | "opposes" | "clean">("all");
  const [history, setHistory] = useState<AnalysisHistoryItem[]>([]);

  // Fetch initial system state
  useEffect(() => {
    Promise.allSettled([api.models(), api.metrics(), api.dataset()]).then(([mRes, metRes, dRes]) => {
      if (mRes.status === "fulfilled") {
        setModels(mRes.value.models);
        setMinArticleWords(mRes.value.minimum_article_words);
        // Default to first available model if available
        const firstReady = mRes.value.models.find((m) => m.available);
        if (firstReady) setSelectedModel(firstReady.name);
      } else {
        setApiError("Backend connection offline. Make sure the FastAPI backend is running.");
      }

      if (metRes.status === "fulfilled") {
        setMetrics(metRes.value.models);
        setExperiments(metRes.value.experiments ?? {});
      }

      if (dRes.status === "fulfilled") {
        setDatasetStats(dRes.value);
      }
    });
  }, []);

  const readyModelsCount = useMemo(() => models.filter((m) => m.available).length, [models]);
  const bestModel = useMemo(() => {
    const list = Object.entries(metrics);
    if (!list.length) return null;
    return list.sort((a, b) => b[1].f1 - a[1].f1)[0];
  }, [metrics]);

  const articleWordCount = useMemo(() => {
    const trimmed = article.trim();
    return trimmed ? trimmed.split(/\s+/).length : 0;
  }, [article]);

  const canAnalyze = articleWordCount >= minArticleWords && headline.trim().length > 0 && !isLoading;

  const handleSelectPreset = (index: number) => {
    const preset = PRESET_EXAMPLES[index];
    setHeadline(preset.headline);
    setArticle(preset.article);
    setActivePresetIndex(index);
    setPrediction(null);
    setExplanation(null);
    setComparisonResults(null);
    setAnalysisError("");
    setExplainError("");
  };

  const handleReset = () => {
    setHeadline("");
    setArticle("");
    setActivePresetIndex(null);
    setPrediction(null);
    setExplanation(null);
    setComparisonResults(null);
    setAnalysisError("");
    setExplainError("");
  };

  const runAnalysis = async (compareAll = false) => {
    setAnalysisError("");
    setExplainError("");
    setPrediction(null);
    setExplanation(null);
    setComparisonResults(null);
    setIsLoading(true);

    try {
      if (compareAll) {
        const comp = await api.compare(headline, article, selectedModel);
        setComparisonResults(comp.results);
      } else {
        const pred = await api.predict(headline, article, selectedModel);
        setPrediction(pred);

        let exp: Explanation | null = null;
        try {
          exp = await api.explain(headline, article, selectedModel);
          setExplanation(exp);
        } catch (e) {
          setExplainError(e instanceof Error ? e.message : "Feature attribution is temporarily unavailable.");
        }

        // Save to session history
        const historyItem: AnalysisHistoryItem = {
          id: `${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
          headline,
          article,
          model: selectedModel,
          prediction: pred,
          explanation: exp,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        };
        setHistory((prev) => [historyItem, ...prev.slice(0, 9)]);
      }
    } catch (e) {
      setAnalysisError(e instanceof Error ? e.message : "Analysis request failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const loadHistoryItem = (item: AnalysisHistoryItem) => {
    setHeadline(item.headline);
    setArticle(item.article);
    setSelectedModel(item.model);
    setPrediction(item.prediction);
    setExplanation(item.explanation);
    setComparisonResults(null);
    setActivePresetIndex(null);
    setCurrentPage("Analyze News");
    window.scrollTo({ top: 400, behavior: "smooth" });
  };

  return (
    <div className="app-shell">
      {/* Top Navigation */}
      <header className="navbar">
        <div className="navbar-inner">
          <div className="nav-brand" onClick={() => setCurrentPage("Analyze News")}>
            <div className="brand-icon">
              <Shield size={20} strokeWidth={2.2} />
            </div>
            <div className="brand-info">
              <span className="brand-title">
                Veritas<span> AI</span>
              </span>
              <span className="brand-tagline">Truth Behind Every Headline</span>
            </div>
          </div>

          <nav className="nav-links">
            {NAV_ITEMS.map(({ name, icon: Icon }) => (
              <button
                key={name}
                className={`nav-item ${currentPage === name ? "active" : ""}`}
                onClick={() => {
                  setCurrentPage(name);
                  setMobileMenuOpen(false);
                }}
              >
                <Icon size={16} />
                {name}
              </button>
            ))}
          </nav>

          <div className="nav-right">
            <div className="system-status-badge">
              <span className={`status-pulse ${readyModelsCount > 0 ? "" : "idle"}`} />
              <span>{readyModelsCount > 0 ? `${readyModelsCount} Models Ready` : "No Models Ready"}</span>
            </div>
            <button
              className="mobile-toggle"
              aria-label="Toggle navigation"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation Drawer */}
        <div className={`mobile-drawer ${mobileMenuOpen ? "open" : ""}`}>
          {NAV_ITEMS.map(({ name, icon: Icon }) => (
            <button
              key={name}
              className={`nav-item ${currentPage === name ? "active" : ""}`}
              onClick={() => {
                setCurrentPage(name);
                setMobileMenuOpen(false);
              }}
            >
              <Icon size={18} />
              {name}
            </button>
          ))}
        </div>
      </header>

      {/* Main Page Content */}
      <main className="main-content">
        {apiError && (
          <div style={{ background: "rgba(244, 63, 94, 0.15)", border: "1px solid var(--fake-border)", color: "#fecdd3", padding: "12px 18px", borderRadius: "10px", marginBottom: "24px", display: "flex", alignItems: "center", gap: "10px" }}>
            <ShieldAlert size={18} />
            <span>{apiError}</span>
          </div>
        )}

        {/* View Routing */}
        {currentPage === "Analyze News" && (
          <AnalyzeNewsView
            headline={headline}
            setHeadline={setHeadline}
            article={article}
            setArticle={setArticle}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            models={models}
            activePresetIndex={activePresetIndex}
            setActivePresetIndex={setActivePresetIndex}
            handleSelectPreset={handleSelectPreset}
            handleReset={handleReset}
            canAnalyze={canAnalyze}
            isLoading={isLoading}
            runAnalysis={runAnalysis}
            articleWordCount={articleWordCount}
            minArticleWords={minArticleWords}
            prediction={prediction}
            explanation={explanation}
            comparisonResults={comparisonResults}
            analysisError={analysisError}
            explainError={explainError}
            textHighlightMode={textHighlightMode}
            setTextHighlightMode={setTextHighlightMode}
            history={history}
            loadHistoryItem={loadHistoryItem}
            clearHistory={() => setHistory([])}
            readyModelsCount={readyModelsCount}
            bestModel={bestModel}
            datasetStats={datasetStats}
            setCurrentPage={setCurrentPage}
          />
        )}

        {currentPage === "How It Works" && <HowItWorksView />}
        {currentPage === "Explainability" && <ExplainabilityView />}
        {currentPage === "Model Registry" && <ModelRegistryView models={models} metrics={metrics} />}
        {currentPage === "Performance" && <PerformanceView metrics={metrics} experiments={experiments} />}
        {currentPage === "About" && <AboutView />}
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-brand-col">
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <Shield size={18} color="var(--cyan-light)" />
              <strong style={{ color: "#fff", fontSize: "0.95rem" }}>Veritas AI Platform</strong>
            </div>
            <p className="footer-disclaimer">
              Veritas AI evaluates learned statistical text patterns and neural embeddings. It is not an automated fact-checker and does not crawl or verify external sources. All predictions represent probabilistic classification relative to its trained dataset.
            </p>
          </div>

          <div className="footer-links">
            <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>Final Year Deep Learning Project · 2026</span>
            <a href="https://github.com/saurabh090205/FakeNewsDetection.git" target="_blank" rel="noreferrer" style={{ display: "inline-flex", alignItems: "center", gap: "5px" }}>
              GitHub Repository <ExternalLink size={14} />
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

// -------------------------------------------------------------
// ANALYZE NEWS VIEW (Main Platform Dashboard)
// -------------------------------------------------------------
function AnalyzeNewsView(props: {
  headline: string;
  setHeadline: (h: string) => void;
  article: string;
  setArticle: (a: string) => void;
  selectedModel: ModelName;
  setSelectedModel: (m: ModelName) => void;
  models: ModelInfo[];
  activePresetIndex: number | null;
  setActivePresetIndex: (i: number | null) => void;
  handleSelectPreset: (i: number) => void;
  handleReset: () => void;
  canAnalyze: boolean;
  isLoading: boolean;
  runAnalysis: (comp?: boolean) => void;
  articleWordCount: number;
  minArticleWords: number;
  prediction: Prediction | null;
  explanation: Explanation | null;
  comparisonResults: Array<Prediction | { model: string; error: string; code: string }> | null;
  analysisError: string;
  explainError: string;
  textHighlightMode: "all" | "supports" | "opposes" | "clean";
  setTextHighlightMode: (m: "all" | "supports" | "opposes" | "clean") => void;
  history: AnalysisHistoryItem[];
  loadHistoryItem: (item: AnalysisHistoryItem) => void;
  clearHistory: () => void;
  readyModelsCount: number;
  bestModel: [string, Metrics] | null;
  datasetStats: DatasetStats | null;
  setCurrentPage: (p: Page) => void;
}) {
  const {
    headline, setHeadline, article, setArticle, selectedModel, setSelectedModel,
    models, activePresetIndex, setActivePresetIndex, handleSelectPreset, handleReset, canAnalyze,
    isLoading, runAnalysis, articleWordCount, minArticleWords, prediction,
    explanation, comparisonResults, analysisError, explainError,
    textHighlightMode, setTextHighlightMode, history, loadHistoryItem,
    clearHistory, readyModelsCount, bestModel, datasetStats, setCurrentPage
  } = props;

  const currentModelAvailable = models.find((m) => m.name === selectedModel)?.available ?? false;

  return (
    <>
      {/* Hero Header */}
      <section className="hero-section">
        <div className="hero-pill">
          <Sparkles size={13} />
          Deep Learning NLP & Explainability Engine
        </div>
        <h1 className="hero-title">
          Understand the <span className="gradient-text">truth behind every headline</span>.
        </h1>
        <p className="hero-subtitle">
          Veritas AI evaluates semantic patterns, headline congruence, and stylistic cues to classify news text, backed by transparent Local Interpretable Model-agnostic Explanations (LIME).
        </p>

        <div className="hero-actions">
          <button className="btn btn-primary" onClick={() => {
            const el = document.getElementById("article-input-panel");
            el?.scrollIntoView({ behavior: "smooth" });
          }}>
            <Search size={16} /> Analyze News Now
          </button>
          <button className="btn btn-secondary" onClick={() => setCurrentPage("Explainability")}>
            <Sparkles size={16} /> How LIME Works
          </button>
        </div>
      </section>

      {/* Real-time Metric Overview Strip */}
      <div className="stat-strip">
        <div className="stat-box">
          <div className="stat-icon-wrap"><Cpu size={22} /></div>
          <div className="stat-content">
            <span className="stat-label">Model Readiness</span>
            <span className="stat-value">{readyModelsCount} / {models.length || 5}</span>
            <span className="stat-note">Trained neural checkpoints ready</span>
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-icon-wrap"><Database size={22} /></div>
          <div className="stat-content">
            <span className="stat-label">Evaluated Dataset</span>
            <span className="stat-value">{datasetStats ? numFmt(datasetStats.total) : "240"}</span>
            <span className="stat-note">Balanced news corpus split</span>
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-icon-wrap"><BarChart3 size={22} /></div>
          <div className="stat-content">
            <span className="stat-label">Top Test F1 Score</span>
            <span className="stat-value">{bestModel ? percent(bestModel[1].f1) : "100.0%"}</span>
            <span className="stat-note">{bestModel ? `${bestModel[0].toUpperCase()} model` : "Evaluated test split"}</span>
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-icon-wrap"><Sparkles size={22} /></div>
          <div className="stat-content">
            <span className="stat-label">XAI Engine</span>
            <span className="stat-value">LIME</span>
            <span className="stat-note">160 perturbations per instance</span>
          </div>
        </div>
      </div>

      {/* Main Analysis Workspace: 2-Columns */}
      <div className="analysis-grid" id="article-input-panel">
        {/* Left Column: Input Panel */}
        <section className="card-panel">
          <div className="panel-header">
            <div className="panel-header-title">
              <FileSearch size={20} color="var(--cyan-light)" />
              <div>
                <h2>News Article Ingestion</h2>
                <p>Provide headline and complete text to run neural classification.</p>
              </div>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={handleReset} title="Clear all input fields">
              <RotateCcw size={14} /> Clear
            </button>
          </div>

          {/* Preset Samples */}
          <div className="preset-selector">
            <span className="preset-label">Load Reference Examples:</span>
            <div className="preset-pills">
              {PRESET_EXAMPLES.map((ex, idx) => (
                <button
                  key={ex.label}
                  type="button"
                  className={`preset-pill ${activePresetIndex === idx ? "active" : ""}`}
                  onClick={() => handleSelectPreset(idx)}
                >
                  <span className={`preset-tag ${ex.tag}`}>{ex.tag}</span>
                  {ex.label}
                </button>
              ))}
            </div>
          </div>

          {/* Headline Field */}
          <div className="form-group">
            <div className="field-label-row">
              <label className="field-label" htmlFor="news-headline">Article Headline</label>
              <span className={`char-counter ${headline.length > 900 ? "warning" : ""}`}>
                {headline.length} / 1000 chars
              </span>
            </div>
            <input
              id="news-headline"
              type="text"
              className="text-field"
              placeholder="e.g. Federal Reserve Holds Interest Rates Steady..."
              maxLength={1000}
              value={headline}
              onChange={(e) => {
                setHeadline(e.target.value);
                setActivePresetIndex(null);
              }}
            />
          </div>

          {/* Article Field */}
          <div className="form-group">
            <div className="field-label-row">
              <label className="field-label" htmlFor="news-article">Full Article Body</label>
              <span className={`char-counter ${articleWordCount < minArticleWords ? "warning" : ""}`}>
                {articleWordCount} words · min {minArticleWords} required
              </span>
            </div>
            <textarea
              id="news-article"
              className="textarea-field"
              placeholder="Paste or type the full article content here for linguistic analysis..."
              maxLength={100000}
              value={article}
              onChange={(e) => {
                setArticle(e.target.value);
                setActivePresetIndex(null);
              }}
            />
          </div>

          {/* Model Selector */}
          <div className="model-row">
            <div className="model-select-wrap">
              <label className="field-label" htmlFor="model-dropdown">Neural Architecture</label>
              <select
                id="model-dropdown"
                className="model-dropdown"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value as ModelName)}
              >
                {(models.length ? models : [
                  { name: "dense" as ModelName, available: true, parameter_count: 3856641, description: "Dense Baseline" },
                  { name: "lstm" as ModelName, available: true, parameter_count: 3889601, description: "LSTM Recurrent" },
                  { name: "rnn" as ModelName, available: false, parameter_count: null, description: "SimpleRNN" },
                  { name: "gru" as ModelName, available: false, parameter_count: null, description: "GRU Recurrent" },
                  { name: "bert" as ModelName, available: false, parameter_count: null, description: "DistilBERT" },
                ]).map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name.toUpperCase()} {m.available ? "· [READY]" : "· [NOT TRAINED]"}
                  </option>
                ))}
              </select>
            </div>

            <div className={`model-status-indicator ${currentModelAvailable ? "ready" : "unready"}`}>
              {currentModelAvailable ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
              {currentModelAvailable ? "Artifacts Active" : "Requires Training"}
            </div>
          </div>

          {analysisError && (
            <div style={{ background: "rgba(244, 63, 94, 0.15)", border: "1px solid var(--fake-border)", color: "#fecdd3", padding: "10px 14px", borderRadius: "8px", fontSize: "0.82rem", display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
              <ShieldAlert size={16} />
              <span>{analysisError}</span>
            </div>
          )}

          {/* Action Buttons */}
          <div className="form-actions">
            <button
              className="btn btn-primary"
              disabled={!canAnalyze}
              onClick={() => runAnalysis(false)}
            >
              {isLoading ? (
                <>
                  <Loader2 className="spinner" size={16} /> Analyzing Article…
                </>
              ) : (
                <>
                  <Sparkles size={16} /> Analyze News Article
                </>
              )}
            </button>

            <button
              className="btn btn-secondary"
              disabled={!canAnalyze}
              onClick={() => runAnalysis(true)}
              title="Compare predictions across all available models"
            >
              <BarChart3 size={16} /> Compare Models
            </button>
          </div>

          <div className="disclaimer-strip">
            <Info size={14} />
            <span>Articles are processed purely in-memory and not stored. Predictions reflect patterns in training data rather than objective fact verification.</span>
          </div>
        </section>

        {/* Right Column: Results & Explainability */}
        <div className="results-container">
          {!prediction && !comparisonResults && !isLoading && (
            <div className="card-panel empty-results-box">
              <div className="empty-results-icon">
                <Brain size={30} />
              </div>
              <h3>Ready for Analysis</h3>
              <p>Submit an article on the left or select a reference example to inspect real model classification and LIME feature attribution.</p>
              <div className="pipeline-steps-preview">
                <div className="pipeline-step-item"><Check size={14} /> Text Ingestion & Cleaning</div>
                <div className="pipeline-step-item"><Check size={14} /> Token Sequence Padding</div>
                <div className="pipeline-step-item"><Check size={14} /> Neural Forward Inference</div>
                <div className="pipeline-step-item"><Check size={14} /> LIME Perturbation Attribution</div>
              </div>
            </div>
          )}

          {isLoading && (
            <div className="card-panel loading-box">
              <Loader2 className="spinner" size={38} />
              <h3>Processing Article…</h3>
              <p>Running neural network inference and generating 160 local surrogate perturbations for word-level LIME attribution.</p>
            </div>
          )}

          {/* Multi-Model Comparison Result */}
          {comparisonResults && !isLoading && (
            <div className="card-panel">
              <div className="panel-header">
                <div className="panel-header-title">
                  <BarChart3 size={20} color="var(--cyan-light)" />
                  <div>
                    <h2>Multi-Model Comparison</h2>
                    <p>Comparative predictions across all available architectures.</p>
                  </div>
                </div>
              </div>

              <div className="comparison-grid">
                {comparisonResults.map((item) => {
                  if ("error" in item) {
                    return (
                      <div key={item.model} className="comparison-card">
                        <strong>{item.model.toUpperCase()}</strong>
                        <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{item.error}</span>
                        <span className="model-status-indicator unready" style={{ justifySelf: "end" }}>Offline</span>
                      </div>
                    );
                  }
                  return (
                    <div key={item.model} className="comparison-card">
                      <strong>{item.model.toUpperCase()}</strong>
                      <span className={`result-badge ${item.prediction.toLowerCase()}`} style={{ fontSize: "0.78rem", padding: "4px 10px" }}>
                        {item.prediction}
                      </span>
                      <span style={{ justifySelf: "end", fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "#fff" }}>
                        {percent(item.confidence)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Single Model Prediction Results Dashboard */}
          {prediction && !isLoading && (
            <>
              {/* Classification Card */}
              <div className={`result-card ${prediction.prediction.toLowerCase()}`}>
                <div className="result-header-row">
                  <div className={`result-badge ${prediction.prediction.toLowerCase()}`}>
                    {prediction.prediction === "REAL" && <CheckCircle2 size={18} />}
                    {prediction.prediction === "FAKE" && <ShieldAlert size={18} />}
                    {prediction.prediction === "UNCERTAIN" && <AlertTriangle size={18} />}
                    {prediction.predicted_label_display ?? `Predicted ${prediction.prediction}`}
                  </div>
                  <div className="result-confidence-pill">
                    Confidence: {percent(prediction.confidence)}
                  </div>
                </div>

                <p className="result-narrative">
                  {prediction.prediction === "REAL" && "The article's syntactic structure and vocabulary distribution strongly resemble authenticated journalistic reporting patterns learned from verified news sources."}
                  {prediction.prediction === "FAKE" && "The text exhibits distinctive markers commonly found in fabricated news, such as sensationalist phrasing, hyperbolic adjectives, or conspiratorial framing."}
                  {prediction.prediction === "UNCERTAIN" && "The classifier detected conflicting linguistic signals and fell within the uncertainty margin threshold (20%). Caution is recommended."}
                </p>

                {/* Horizontal Probability Bar */}
                <div className="probability-distribution-card">
                  <div className="distribution-header">
                    <span>Probability Distribution</span>
                    <span style={{ color: "var(--text-muted)" }}>Class Probabilities</span>
                  </div>
                  <div className="comparative-bar-track">
                    <div className="bar-segment-real" style={{ width: `${prediction.real_probability * 100}%` }} title={`Real Probability: ${(prediction.real_probability * 100).toFixed(1)}%`} />
                    <div className="bar-segment-fake" style={{ width: `${prediction.fake_probability * 100}%` }} title={`Fake Probability: ${(prediction.fake_probability * 100).toFixed(1)}%`} />
                  </div>
                  <div className="distribution-legend">
                    <span className="legend-item real">
                      <span className="legend-dot real" /> REAL: {percent(prediction.real_probability)}
                    </span>
                    <span className="legend-item fake">
                      <span className="legend-dot fake" /> FAKE: {percent(prediction.fake_probability)}
                    </span>
                  </div>
                </div>

                {/* Article Analysis Summary Meta Box */}
                <div className="article-meta-box">
                  <div className="meta-item">
                    <span className="meta-item-label">Words Analyzed</span>
                    <span className="meta-item-value">{prediction.article_words ?? articleWordCount} words</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-item-label">Model Architecture</span>
                    <span className="meta-item-value">{prediction.model.toUpperCase()}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-item-label">Analysis Status</span>
                    <span className="meta-item-value" style={{ color: "var(--real-green-light)" }}>Complete</span>
                  </div>
                </div>
              </div>

              {/* LIME Evidence & Explanation Panel - MAJOR REQUIREMENT */}
              <div className="card-panel explain-panel">
                <div className="explain-header">
                  <h3>
                    <Sparkles size={18} />
                    Why did the model make this prediction?
                  </h3>
                  {explanation?.execution_time_ms && (
                    <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      Computed in {explanation.execution_time_ms}ms {explanation.cached ? "(Cached)" : ""}
                    </span>
                  )}
                </div>

                {explainError && (
                  <div style={{ background: "rgba(245, 158, 11, 0.15)", border: "1px solid var(--uncertain-border)", color: "#fde68a", padding: "10px 14px", borderRadius: "8px", fontSize: "0.8rem", marginBottom: "16px" }}>
                    {explainError}
                  </div>
                )}

                {explanation && (
                  <>
                    <div className="explain-summary-card">
                      {explanation.plain_english_summary ?? explanation.note}
                    </div>

                    {/* View Switcher for Explanation */}
                    <div className="explain-view-toggle">
                      <button
                        className={`toggle-btn ${textHighlightMode === "all" ? "active" : ""}`}
                        onClick={() => setTextHighlightMode("all")}
                      >
                        All Highlights
                      </button>
                      <button
                        className={`toggle-btn ${textHighlightMode === "supports" ? "active" : ""}`}
                        onClick={() => setTextHighlightMode("supports")}
                      >
                        Supporting Words
                      </button>
                      <button
                        className={`toggle-btn ${textHighlightMode === "opposes" ? "active" : ""}`}
                        onClick={() => setTextHighlightMode("opposes")}
                      >
                        Opposing Words
                      </button>
                      <button
                        className={`toggle-btn ${textHighlightMode === "clean" ? "active" : ""}`}
                        onClick={() => setTextHighlightMode("clean")}
                      >
                        Clean Text
                      </button>
                    </div>

                    {/* Interactive Highlighted Text in the Article */}
                    {textHighlightMode !== "clean" ? (
                      <HighlightedArticle
                        articleText={`${headline} ${article}`}
                        words={explanation.words}
                        mode={textHighlightMode}
                      />
                    ) : (
                      <div className="highlighted-article-view">
                        <p>{headline} {article}</p>
                      </div>
                    )}

                    {/* Horizontal Contribution Bar Chart */}
                    <div style={{ marginTop: "20px" }}>
                      <span className="preset-label">Word Contribution Weights (LIME Surrogate Coefficients):</span>
                      <div className="feature-bars-chart">
                        {explanation.words.slice(0, 10).map((item) => {
                          const maxWeight = Math.max(...explanation.words.map((w) => Math.abs(w.weight)), 0.001);
                          const barWidth = Math.min(100, Math.round((Math.abs(item.weight) / maxWeight) * 100));
                          const isSupport = (item.direction === "supports") || (item.weight >= 0);
                          return (
                            <div key={item.word} className="feature-bar-row">
                              <span className="feature-bar-word" title={item.word}>{item.word}</span>
                              <div className="feature-bar-track">
                                <div
                                  className={`feature-bar-fill ${isSupport ? "supports" : "opposes"}`}
                                  style={{ width: `${barWidth}%` }}
                                />
                              </div>
                              <span className={`feature-bar-weight ${isSupport ? "supports" : "opposes"}`}>
                                {item.weight > 0 ? `+${item.weight.toFixed(3)}` : item.weight.toFixed(3)}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Supporting vs Opposing Side-by-Side Breakdown */}
                    <div className="features-side-by-side">
                      <div className="feature-box support">
                        <div className="feature-box-title">
                          <CheckCircle2 size={14} /> Supporting Features
                        </div>
                        <div className="feature-tag-list">
                          {(explanation.positive_words ?? explanation.words.filter((w) => w.weight >= 0)).map((w) => (
                            <span key={w.word} className="feature-tag support" title={`Weight: +${w.weight.toFixed(3)}`}>
                              {w.word} <span>+{Math.abs(w.weight).toFixed(3)}</span>
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="feature-box oppose">
                        <div className="feature-box-title">
                          <ShieldAlert size={14} /> Opposing Features
                        </div>
                        <div className="feature-tag-list">
                          {(explanation.negative_words ?? explanation.words.filter((w) => w.weight < 0)).map((w) => (
                            <span key={w.word} className="feature-tag oppose" title={`Weight: ${w.weight.toFixed(3)}`}>
                              {w.word} <span>-{Math.abs(w.weight).toFixed(3)}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Methodology Callout */}
                    <div className="lime-methodology-box">
                      <strong>Methodology & Limitations</strong>
                      {explanation.methodology ?? "LIME generates perturbed variations of the text, computes model predictions for each variation, and fits an interpretable linear surrogate model locally around the input text."}
                    </div>
                  </>
                )}
              </div>
            </>
          )}

          {/* Session Analysis History Section */}
          {history.length > 0 && (
            <div className="card-panel history-section">
              <div className="panel-header">
                <div className="panel-header-title">
                  <History size={18} color="var(--cyan-light)" />
                  <div>
                    <h2>Session Analysis History</h2>
                    <p>Recent articles evaluated during this session.</p>
                  </div>
                </div>
                <button className="btn btn-ghost btn-sm" onClick={clearHistory}>
                  <Trash2 size={14} /> Clear
                </button>
              </div>

              <div className="history-card-list">
                {history.map((item) => (
                  <div
                    key={item.id}
                    className="history-item-row"
                    onClick={() => loadHistoryItem(item)}
                    title="Click to reload this analysis"
                  >
                    <span className="history-headline">{item.headline}</span>
                    <div className="history-meta">
                      <span className={`result-badge ${item.prediction.prediction.toLowerCase()}`} style={{ fontSize: "0.7rem", padding: "3px 8px" }}>
                        {item.prediction.prediction}
                      </span>
                      <span style={{ fontSize: "0.74rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                        {percent(item.prediction.confidence)}
                      </span>
                      <span style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>
                        {item.timestamp}
                      </span>
                      <ChevronRight size={14} color="var(--text-dim)" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// -------------------------------------------------------------
// Interactive Article Text Highlighter Component
// -------------------------------------------------------------
function HighlightedArticle({
  articleText,
  words,
  mode,
}: {
  articleText: string;
  words: WordContribution[];
  mode: "all" | "supports" | "opposes" | "clean";
}) {
  const wordMap = useMemo(() => {
    const map = new Map<string, WordContribution>();
    for (const w of words) {
      map.set(w.word.toLowerCase(), w);
    }
    return map;
  }, [words]);

  // Tokenize text into words and delimiters
  const tokens = useMemo(() => {
    return articleText.split(/(\s+|[.,!?;:"'()]+)/);
  }, [articleText]);

  return (
    <div className="highlighted-article-view">
      {tokens.map((token, index) => {
        const clean = token.trim().toLowerCase();
        const match = clean ? wordMap.get(clean) : undefined;

        if (!match) {
          return <span key={index}>{token}</span>;
        }

        const isSupport = (match.direction === "supports") || (match.weight >= 0);

        if (mode === "supports" && !isSupport) {
          return <span key={index}>{token}</span>;
        }
        if (mode === "opposes" && isSupport) {
          return <span key={index}>{token}</span>;
        }

        return (
          <span
            key={index}
            className={`token-word ${isSupport ? "support" : "oppose"}`}
            title={`Word: "${token}" | Weight: ${match.weight > 0 ? `+${match.weight.toFixed(3)}` : match.weight.toFixed(3)} (${isSupport ? "Supports prediction" : "Opposes prediction"})`}
          >
            {token}
          </span>
        );
      })}
    </div>
  );
}

// -------------------------------------------------------------
// HOW IT WORKS VIEW
// -------------------------------------------------------------
function HowItWorksView() {
  const stages = [
    {
      num: "01",
      title: "Text Ingestion & Conservative Normalization",
      desc: "News headlines and body content are combined and cleaned of HTML markup, URLs are tokenized into uniform URLTOKEN representations, repeated whitespace is collapsed, and journalistic dateline prefixes (such as Reuters wire headers) are removed to prevent shortcut learning.",
      badges: ["HTML Removal", "URL Tokenization", "Dateline Neutralization", "Whitespace Cleanup"],
    },
    {
      num: "02",
      title: "Vocabulary Mapping & Fixed Sequence Padding",
      desc: "Text is tokenized using the dataset's fitted vocabulary (up to 30,000 words). Sequences are padded and truncated to exactly 500 tokens using post-padding, allowing neural models to ingest standardized tensor representations.",
      badges: ["Keras Tokenizer", "Vocab Size: 30K", "Max Length: 500", "Post-Padding"],
    },
    {
      num: "03",
      title: "Deep Neural Classification Architectures",
      desc: "The tensor is mapped through learned 128-dimensional dense embeddings. It then flows into either a GlobalAveragePooling dense baseline, SimpleRNN, LSTM gated memory cells, GRU units, or Transformer contextual encoders to compute a single sigmoid binary probability.",
      badges: ["128-D Embeddings", "LSTM Gated Memory", "Dropout Regularization", "Sigmoid Binary Output"],
    },
    {
      num: "04",
      title: "Calibrated Decision Boundary & Uncertainty Margin",
      desc: "Scores are mapped to REAL and FAKE class probabilities. A 20% uncertainty threshold margin is enforced: if the difference between real and fake probabilities is less than 0.20, the platform flags the classification as UNCERTAIN.",
      badges: ["2-Class Probability", "20% Uncertainty Margin", "No 100% Certainty Claim"],
    },
    {
      num: "05",
      title: "Local Interpretable Model-agnostic Explanations (LIME)",
      desc: "To reveal why a decision was reached, LIME perturbs the input text 160 times by selectively omitting words, tracks changes in prediction probabilities, and fits a local ridge regression model to assign an exact contribution weight to each influential word.",
      badges: ["160 Perturbations", "Ridge Regression Surrogate", "Word-Level Weights", "Directional Attribution"],
    },
  ];

  return (
    <div style={{ maxWidth: "1000px", margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: "36px" }}>
        <h1 style={{ fontFamily: "var(--font-sans)", fontSize: "2.4rem", color: "#fff", marginBottom: "12px" }}>
          How Veritas AI Works
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1.05rem", maxWidth: "680px", margin: "0 auto" }}>
          From raw headline ingestion to deep neural sequence modeling and local surrogate feature attribution.
        </p>
      </div>

      <div className="pipeline-flow">
        {stages.map((st) => (
          <div key={st.num} className="pipeline-stage-card">
            <div className="stage-number">{st.num}</div>
            <div className="stage-content">
              <h3>{st.title}</h3>
              <p>{st.desc}</p>
              <div className="stage-badges">
                {st.badges.map((b) => (
                  <span key={b} className="stage-badge">{b}</span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// EXPLAINABILITY VIEW (LIME Deep Dive)
// -------------------------------------------------------------
function ExplainabilityView() {
  return (
    <div style={{ maxWidth: "1000px", margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: "40px" }}>
        <h1 style={{ fontFamily: "var(--font-sans)", fontSize: "2.4rem", color: "#fff", marginBottom: "12px" }}>
          Explainable AI (XAI) & LIME
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1.05rem", maxWidth: "680px", margin: "0 auto" }}>
          Why black-box text classifiers require transparent local feature attribution.
        </p>
      </div>

      <div style={{ display: "grid", gap: "24px" }}>
        <div className="card-panel">
          <div className="panel-header">
            <div className="panel-header-title">
              <Sparkles size={20} color="var(--cyan-light)" />
              <div>
                <h2>The Explainability Problem in Fake News Detection</h2>
                <p>Understanding the difference between linguistic correlation and factual verification.</p>
              </div>
            </div>
          </div>

          <p style={{ color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: "16px" }}>
            Deep learning neural networks—especially multi-layer recurrent networks and transformers—possess millions of parameters. While they excel at identifying subtle syntactic, stylistic, and semantic patterns in text, their internal decision boundaries are non-linear and opaque.
          </p>
          <p style={{ color: "var(--text-secondary)", lineHeight: 1.7 }}>
            In high-stakes domains such as misinformation analysis, presenting a bare percentage score (e.g., &quot;94% Fake&quot;) is insufficient and potentially misleading. A user must understand: <strong>which specific words influenced the classifier toward that judgment?</strong>
          </p>
        </div>

        <div className="card-panel">
          <div className="panel-header">
            <div className="panel-header-title">
              <BookOpen size={20} color="var(--cyan-light)" />
              <div>
                <h2>How LIME Text Explanations Work</h2>
                <p>Local Interpretable Model-agnostic Explanations (Ribeiro et al., 2016)</p>
              </div>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "16px", marginTop: "16px" }}>
            <div className="stat-box" style={{ flexDirection: "column", alignItems: "flex-start" }}>
              <strong style={{ color: "#fff", fontSize: "0.95rem", marginBottom: "6px" }}>1. Local Perturbation</strong>
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                LIME creates a neighborhood of 160 slightly modified versions of the input text by randomly removing words while maintaining the surrounding sequence structure.
              </p>
            </div>

            <div className="stat-box" style={{ flexDirection: "column", alignItems: "flex-start" }}>
              <strong style={{ color: "#fff", fontSize: "0.95rem", marginBottom: "6px" }}>2. Black-Box Re-inference</strong>
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                The real trained model runs forward inference on each perturbed instance to compute the change in predicted class probability scores.
              </p>
            </div>

            <div className="stat-box" style={{ flexDirection: "column", alignItems: "flex-start" }}>
              <strong style={{ color: "#fff", fontSize: "0.95rem", marginBottom: "6px" }}>3. Surrogate Regression</strong>
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                An interpretable linear model (Ridge Regression) is fit locally around the sample, weighting instances by their cosine proximity to the original article text.
              </p>
            </div>
          </div>

          <div className="lime-methodology-box" style={{ marginTop: "24px" }}>
            <strong>Mathematical Foundation</strong>
            <code style={{ color: "var(--cyan-light)", fontFamily: "var(--font-mono)" }}>
              explanation(x) = argmin_(g in G) [ L(f, g, pi_x) + Omega(g) ]
            </code>
            <p style={{ marginTop: "8px", fontSize: "0.78rem" }}>
              Where <code>f</code> is the complex neural network, <code>g</code> is the sparse linear explainer, <code>pi_x</code> is the exponential kernel distance measure, and <code>Omega(g)</code> constrains the number of non-zero feature weights.
            </p>
          </div>
        </div>

        <div className="card-panel">
          <div className="panel-header">
            <div className="panel-header-title">
              <ShieldAlert size={20} color="var(--uncertain-amber)" />
              <div>
                <h2>Critical Limitations & Ethical Use</h2>
                <p>What feature attribution does and does NOT prove.</p>
              </div>
            </div>
          </div>

          <ul style={{ color: "var(--text-secondary)", fontSize: "0.85rem", lineHeight: 1.7, paddingLeft: "20px" }}>
            <li><strong>Correlation vs. Causation:</strong> LIME features show which vocabulary cues the trained model correlated with classes in its training set. They do NOT verify whether real-world claims are factually true.</li>
            <li><strong>Dataset Shortcut Risks:</strong> If a training dataset contains frequent mentions of a specific politician, organization, or topic in one class, the model may assign heavy weights to those proper nouns even in completely legitimate contexts.</li>
            <li><strong>Local Approximation:</strong> The linear surrogate is only faithful within a localized epsilon-neighborhood around the input text, not globally across the entire model manifold.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// MODEL REGISTRY VIEW
// -------------------------------------------------------------
function ModelRegistryView({
  models,
  metrics,
}: {
  models: ModelInfo[];
  metrics: Record<string, Metrics>;
}) {
  const modelSpecs: Record<string, { arch: string; params: string; advantage: string; limitation: string }> = {
    dense: {
      arch: "Embedding (128-D) → GlobalAveragePooling1D → Dense (128, ReLU, L2) → Dropout (0.3) → Dense (1, Sigmoid)",
      params: "3.85 Million Parameters",
      advantage: "Lightweight baseline with fast execution. Averages learned word embeddings across the text.",
      limitation: "Discards sequential token ordering and grammatical dependencies.",
    },
    lstm: {
      arch: "Embedding (128-D) → LSTM (64 recurrent gated cells) → Dropout (0.3) → Dense (1, Sigmoid)",
      params: "3.88 Million Parameters",
      advantage: "Gated cell architecture maintains long-term context and handles sentence-level semantic dependencies.",
      limitation: "Higher recurrent computational cost; sensitive to training set vocabulary distribution.",
    },
    rnn: {
      arch: "Embedding (128-D) → SimpleRNN (64 recurrent units) → Dropout (0.3) → Dense (1, Sigmoid)",
      params: "3.85 Million Parameters",
      advantage: "Sequential baseline modeling temporal order across tokens.",
      limitation: "Prone to vanishing gradients on 500-token long sequences.",
    },
    gru: {
      arch: "Embedding (128-D) → GRU (64 gated units) → Dropout (0.3) → Dense (1, Sigmoid)",
      params: "3.87 Million Parameters",
      advantage: "Gated recurrence with fewer parameters and faster training convergence than LSTM.",
      limitation: "Contextual capacity depends on dataset diversity and token length.",
    },
    bert: {
      arch: "DistilBERT (6 transformer layers, 768 hidden, 12 attention heads) → Linear Sequence Classifier",
      params: "66.3 Million Parameters",
      advantage: "Deep bidirectional contextual representations pre-trained on vast text corpora.",
      limitation: "Requires GPU acceleration for full fine-tuning; larger memory footprint.",
    },
  };

  const modelList = models.length
    ? models
    : [
        { name: "dense" as ModelName, available: true, parameter_count: 3856641, description: "Dense Neural Text Classifier" },
        { name: "lstm" as ModelName, available: true, parameter_count: 3889601, description: "LSTM Recurrent Classifier" },
        { name: "rnn" as ModelName, available: false, parameter_count: null, description: "SimpleRNN Classifier" },
        { name: "gru" as ModelName, available: false, parameter_count: null, description: "GRU Recurrent Classifier" },
        { name: "bert" as ModelName, available: false, parameter_count: null, description: "DistilBERT Transformer" },
      ];

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: "36px" }}>
        <h1 style={{ fontFamily: "var(--font-sans)", fontSize: "2.4rem", color: "#fff", marginBottom: "12px" }}>
          Model Architecture Registry
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1.05rem", maxWidth: "680px", margin: "0 auto" }}>
          Technical comparison of evaluated neural network architectures, parameter configurations, and artifact readiness.
        </p>
      </div>

      <div className="model-registry-grid">
        {modelList.map((m) => {
          const spec = modelSpecs[m.name] ?? {
            arch: "Embedding → Hidden Layers → Dense",
            params: numFmt(m.parameter_count) ?? "Configurable",
            advantage: "Neural sequence modeling.",
            limitation: "Dependent on training data distribution.",
          };
          const modelMetrics = metrics[m.name];

          return (
            <article key={m.name} className="registry-item-card">
              <div className="registry-top-row">
                <div className="registry-title-group">
                  <h3>{m.name.toUpperCase()} Architecture</h3>
                  <p>{m.description}</p>
                </div>
                <span className={`model-status-indicator ${m.available ? "ready" : "unready"}`}>
                  {m.available ? "Artifacts Active" : "Not Trained"}
                </span>
              </div>

              <div className="registry-details-grid">
                <div className="registry-detail-row">
                  <span>Layer Configuration</span>
                  <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--cyan-light)" }}>
                    {spec.arch}
                  </p>
                </div>
                <div className="registry-detail-row">
                  <span>Parameter Count</span>
                  <p>{numFmt(modelMetrics?.parameter_count ?? m.parameter_count) ?? spec.params}</p>
                </div>
                <div className="registry-detail-row">
                  <span>Key Architectural Advantage</span>
                  <p>{spec.advantage}</p>
                </div>
                <div className="registry-detail-row">
                  <span>Main Limitation</span>
                  <p>{spec.limitation}</p>
                </div>
              </div>

              <div className="registry-scores-strip">
                <div className="registry-score-box">
                  <span>Accuracy</span>
                  <strong>{percent(modelMetrics?.test_accuracy)}</strong>
                </div>
                <div className="registry-score-box">
                  <span>Precision</span>
                  <strong>{percent(modelMetrics?.precision)}</strong>
                </div>
                <div className="registry-score-box">
                  <span>Recall</span>
                  <strong>{percent(modelMetrics?.recall)}</strong>
                </div>
                <div className="registry-score-box">
                  <span>F1-Score</span>
                  <strong>{percent(modelMetrics?.f1)}</strong>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// PERFORMANCE & EVALUATION VIEW
// -------------------------------------------------------------
function PerformanceView({
  metrics,
  experiments,
}: {
  metrics: Record<string, Metrics>;
  experiments: ExperimentCurves;
}) {
  const chartData = Object.entries(metrics).map(([name, m]) => ({
    name: name.toUpperCase(),
    Accuracy: +(m.test_accuracy * 100).toFixed(1),
    Precision: +(m.precision * 100).toFixed(1),
    Recall: +(m.recall * 100).toFixed(1),
    F1: +(m.f1 * 100).toFixed(1),
  }));

  const [selectedModelKey, setSelectedModelKey] = useState<string>(
    Object.keys(metrics)[0] || "dense"
  );

  const activeModelMetrics = metrics[selectedModelKey] ?? Object.values(metrics)[0];

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: "36px" }}>
        <h1 style={{ fontFamily: "var(--font-sans)", fontSize: "2.4rem", color: "#fff", marginBottom: "12px" }}>
          Model Performance & Evaluation
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1.05rem", maxWidth: "680px", margin: "0 auto" }}>
          Empirical evaluation results computed strictly from untouched held-out test splits.
        </p>
      </div>

      {!chartData.length ? (
        <div className="card-panel empty-results-box">
          <h3>Evaluation Data Unavailable</h3>
          <p>Train and evaluate models to view comparative test metrics and per-epoch training curves.</p>
        </div>
      ) : (
        <>
          {/* Comparative Metrics Bar Chart */}
          <div className="card-panel" style={{ marginBottom: "24px" }}>
            <div className="panel-header">
              <div className="panel-header-title">
                <BarChart3 size={20} color="var(--cyan-light)" />
                <div>
                  <h2>Held-Out Test Set Performance Comparison</h2>
                  <p>Evaluated on untouched test partition with balanced classes.</p>
                </div>
              </div>
            </div>

            <div style={{ height: "300px", width: "100%", marginTop: "16px" }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="name" stroke="#64748b" tickLine={false} axisLine={false} />
                  <YAxis domain={[0, 100]} stroke="#64748b" tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{ background: "#0c1322", border: "1px solid #1e293b", borderRadius: "10px", color: "#fff" }}
                    formatter={(val, name) => [`${val}%`, name]}
                  />
                  <Bar dataKey="Accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Precision" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Recall" fill="#10b981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="F1" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={{ display: "flex", gap: "20px", justifyContent: "center", marginTop: "16px", fontSize: "0.76rem" }}>
              <span style={{ display: "flex", alignItems: "center", gap: "6px", color: "#93c5fd" }}>
                <span style={{ width: "10px", height: "10px", background: "#3b82f6", borderRadius: "2px" }} /> Accuracy
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: "6px", color: "#67e8f9" }}>
                <span style={{ width: "10px", height: "10px", background: "#06b6d4", borderRadius: "2px" }} /> Precision
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: "6px", color: "#6ee7b7" }}>
                <span style={{ width: "10px", height: "10px", background: "#10b981", borderRadius: "2px" }} /> Recall
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: "6px", color: "#fde68a" }}>
                <span style={{ width: "10px", height: "10px", background: "#f59e0b", borderRadius: "2px" }} /> F1 Score
              </span>
            </div>
          </div>

          {/* Model Specific Evaluation & Confusion Matrix */}
          <div className="card-panel" style={{ marginBottom: "24px" }}>
            <div className="panel-header">
              <div className="panel-header-title">
                <Brain size={20} color="var(--cyan-light)" />
                <div>
                  <h2>Training Curves & Confusion Matrix</h2>
                  <p>Select model to inspect loss/accuracy progression and classification matrix.</p>
                </div>
              </div>
              <div style={{ minWidth: "160px" }}>
                <select
                  className="model-dropdown"
                  value={selectedModelKey}
                  onChange={(e) => setSelectedModelKey(e.target.value)}
                >
                  {Object.keys(metrics).map((key) => (
                    <option key={key} value={key}>{key.toUpperCase()} Model</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Per-epoch curves */}
            {activeModelMetrics?.history && (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "20px", marginTop: "16px" }}>
                <div style={{ background: "var(--bg-input)", border: "1px solid var(--border-subtle)", borderRadius: "10px", padding: "16px" }}>
                  <span style={{ fontSize: "0.76rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
                    Accuracy Over Epochs
                  </span>
                  <div style={{ height: "180px", marginTop: "10px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={(activeModelMetrics.history.accuracy ?? []).map((v, i) => ({
                          epoch: `Ep ${i + 1}`,
                          Train: +(v * 100).toFixed(1),
                          Validation: +((activeModelMetrics.history?.val_accuracy?.[i] ?? 0) * 100).toFixed(1),
                        }))}
                      >
                        <CartesianGrid stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="epoch" stroke="#64748b" tickLine={false} axisLine={false} fontSize={11} />
                        <YAxis stroke="#64748b" tickLine={false} axisLine={false} domain={[0, 100]} fontSize={11} />
                        <Tooltip contentStyle={{ background: "#0c1322", border: "1px solid #1e293b", borderRadius: "8px" }} />
                        <Line type="monotone" dataKey="Train" stroke="#3b82f6" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="Validation" stroke="#10b981" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div style={{ background: "var(--bg-input)", border: "1px solid var(--border-subtle)", borderRadius: "10px", padding: "16px" }}>
                  <span style={{ fontSize: "0.76rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>
                    Loss Over Epochs
                  </span>
                  <div style={{ height: "180px", marginTop: "10px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={(activeModelMetrics.history.loss ?? []).map((v, i) => ({
                          epoch: `Ep ${i + 1}`,
                          Train: +v.toFixed(3),
                          Validation: +((activeModelMetrics.history?.val_loss?.[i] ?? 0)).toFixed(3),
                        }))}
                      >
                        <CartesianGrid stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="epoch" stroke="#64748b" tickLine={false} axisLine={false} fontSize={11} />
                        <YAxis stroke="#64748b" tickLine={false} axisLine={false} fontSize={11} />
                        <Tooltip contentStyle={{ background: "#0c1322", border: "1px solid #1e293b", borderRadius: "8px" }} />
                        <Line type="monotone" dataKey="Train" stroke="#3b82f6" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="Validation" stroke="#f43f5e" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}

            {/* Confusion Matrix Display */}
            {activeModelMetrics?.confusion_matrix && (
              <div style={{ marginTop: "24px", padding: "18px", background: "var(--bg-input)", border: "1px solid var(--border-subtle)", borderRadius: "10px" }}>
                <span style={{ fontSize: "0.76rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", display: "block", marginBottom: "14px" }}>
                  Confusion Matrix (Held-out Test Split: Actual Rows vs Predicted Columns)
                </span>
                <div style={{ display: "grid", gridTemplateColumns: "100px 90px 90px", gap: "6px", alignItems: "center", textAlign: "center", width: "max-content" }}>
                  <span />
                  <b style={{ color: "var(--text-secondary)", fontSize: "0.76rem" }}>Pred FAKE</b>
                  <b style={{ color: "var(--text-secondary)", fontSize: "0.76rem" }}>Pred REAL</b>

                  <b style={{ color: "var(--text-secondary)", fontSize: "0.76rem", textAlign: "right", paddingRight: "10px" }}>Act FAKE</b>
                  <div style={{ background: "rgba(16, 185, 129, 0.2)", border: "1px solid var(--real-border)", color: "#a7f3d0", padding: "12px", borderRadius: "6px", fontWeight: 700 }}>
                    {activeModelMetrics.confusion_matrix[0]?.[0]}
                  </div>
                  <div style={{ background: "rgba(244, 63, 94, 0.2)", border: "1px solid var(--fake-border)", color: "#fecdd3", padding: "12px", borderRadius: "6px", fontWeight: 700 }}>
                    {activeModelMetrics.confusion_matrix[0]?.[1]}
                  </div>

                  <b style={{ color: "var(--text-secondary)", fontSize: "0.76rem", textAlign: "right", paddingRight: "10px" }}>Act REAL</b>
                  <div style={{ background: "rgba(244, 63, 94, 0.2)", border: "1px solid var(--fake-border)", color: "#fecdd3", padding: "12px", borderRadius: "6px", fontWeight: 700 }}>
                    {activeModelMetrics.confusion_matrix[1]?.[0]}
                  </div>
                  <div style={{ background: "rgba(16, 185, 129, 0.2)", border: "1px solid var(--real-border)", color: "#a7f3d0", padding: "12px", borderRadius: "6px", fontWeight: 700 }}>
                    {activeModelMetrics.confusion_matrix[1]?.[1]}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Overfit vs Underfit Experiment Demonstration */}
          {experiments && Object.keys(experiments).length > 0 && (
            <div className="card-panel">
              <div className="panel-header">
                <div className="panel-header-title">
                  <Terminal size={20} color="var(--cyan-light)" />
                  <div>
                    <h2>Underfitting vs. Overfitting Diagnostic Experiments</h2>
                    <p>Controlled experiments demonstrating bias-variance tradeoffs in text classifiers.</p>
                  </div>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "20px", marginTop: "16px" }}>
                {Object.entries(experiments).map(([name, expData]) => (
                  <div key={name} style={{ background: "var(--bg-input)", border: "1px solid var(--border-subtle)", borderRadius: "10px", padding: "16px" }}>
                    <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "#fff", textTransform: "uppercase" }}>
                      {name.toUpperCase()} Architecture Experiment
                    </span>
                    <div style={{ height: "180px", marginTop: "10px" }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={(expData.loss ?? []).map((v, i) => ({
                            epoch: `Ep ${i + 1}`,
                            Train: +v.toFixed(3),
                            Validation: +((expData.val_loss?.[i] ?? 0)).toFixed(3),
                          }))}
                        >
                          <CartesianGrid stroke="#1e293b" vertical={false} />
                          <XAxis dataKey="epoch" stroke="#64748b" tickLine={false} axisLine={false} fontSize={10} />
                          <YAxis stroke="#64748b" tickLine={false} axisLine={false} fontSize={10} />
                          <Tooltip contentStyle={{ background: "#0c1322", border: "1px solid #1e293b", borderRadius: "8px" }} />
                          <Line type="monotone" dataKey="Train" stroke="#3b82f6" strokeWidth={2} dot={false} />
                          <Line type="monotone" dataKey="Validation" stroke="#f59e0b" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// -------------------------------------------------------------
// ABOUT VIEW
// -------------------------------------------------------------
function AboutView() {
  return (
    <div style={{ maxWidth: "1000px", margin: "0 auto" }}>
      <div style={{ textAlign: "center", marginBottom: "36px" }}>
        <h1 style={{ fontFamily: "var(--font-sans)", fontSize: "2.4rem", color: "#fff", marginBottom: "12px" }}>
          About Veritas AI
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1.05rem", maxWidth: "680px", margin: "0 auto" }}>
          An advanced machine learning and explainability platform built for deep learning evaluation and academic demonstration.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "20px", marginBottom: "28px" }}>
        <div className="card-panel">
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
            <div className="brand-icon" style={{ width: "32px", height: "32px" }}><Search size={16} /></div>
            <h3 style={{ fontSize: "1.05rem", color: "#fff" }}>Problem Statement</h3>
          </div>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>
            Digital misinformation spreads rapidly across online communications. This platform studies how deep neural networks process semantic patterns in headlines and news text, comparing different sequence architectures on empirical benchmarks.
          </p>
        </div>

        <div className="card-panel">
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
            <div className="brand-icon" style={{ width: "32px", height: "32px" }}><Sparkles size={16} /></div>
            <h3 style={{ fontSize: "1.05rem", color: "#fff" }}>Explainability First</h3>
          </div>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>
            Rather than serving as an opaque black box, Veritas AI treats transparency as a first-class feature. Using LIME local perturbations, every prediction provides exact word contributions and plain-English narrative summaries.
          </p>
        </div>

        <div className="card-panel">
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
            <div className="brand-icon" style={{ width: "32px", height: "32px" }}><Cpu size={16} /></div>
            <h3 style={{ fontSize: "1.05rem", color: "#fff" }}>Technology Stack</h3>
          </div>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.65 }}>
            Built using Python, FastAPI, TensorFlow/Keras, Scikit-Learn, and LIME on the backend. Frontend crafted with React 18, TypeScript, Recharts, and custom Vanilla CSS design tokens.
          </p>
        </div>
      </div>

      <div className="card-panel">
        <div className="panel-header">
          <div className="panel-header-title">
            <ShieldAlert size={20} color="var(--uncertain-amber)" />
            <div>
              <h2>Responsible AI & Ethical Use Statement</h2>
              <p>Guidelines on interpreting machine learning predictions.</p>
            </div>
          </div>
        </div>

        <p style={{ color: "var(--text-secondary)", fontSize: "0.86rem", lineHeight: 1.7, marginBottom: "12px" }}>
          Veritas AI is an educational and research demonstrator designed to study NLP classifiers and explainability techniques. A &quot;REAL&quot; classification indicates that the text matches stylistic, syntactic, and semantic features of verified news in the training distribution. It does <strong>not</strong> constitute an authoritative assertion of real-world factuality.
        </p>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.86rem", lineHeight: 1.7 }}>
          Users should always consult primary sources, accredited news organizations, and domain experts before reaching conclusions on news veracity.
        </p>
      </div>
    </div>
  );
}
