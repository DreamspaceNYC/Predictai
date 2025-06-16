import React, { useState, useEffect, createContext, useContext } from 'react';
import { BrowserRouter, Routes, Route, Navigate, Link } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';
import axios from 'axios';
import { 
  TrendingUp, 
  Shield, 
  Upload, 
  RefreshCw, 
  BarChart3, 
  LogOut, 
  Mail,
  Eye,
  EyeOff,
  Activity,
  Target,
  Calendar,
  Users
} from 'lucide-react';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Supabase client
const supabase = createClient(
  process.env.REACT_APP_SUPABASE_URL,
  process.env.REACT_APP_SUPABASE_ANON_KEY
);

// Auth Context
const AuthContext = createContext();

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      setLoading(false);
    });

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      setUser(session?.user ?? null);
      setLoading(false);
    });

    return () => subscription.unsubscribe();
  }, []);

  const signIn = async (email) => {
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: {
        emailRedirectTo: window.location.origin,
      },
    });
    return { error };
  };

  const signOut = async () => {
    await supabase.auth.signOut();
  };

  const isAdmin = () => {
    // For demo purposes, admin emails end with @admin.com
    return user?.email?.endsWith('@admin.com') || user?.email === 'admin@predictbet.ai';
  };

  return (
    <AuthContext.Provider value={{
      user,
      session,
      loading,
      signIn,
      signOut,
      isAdmin
    }}>
      {children}
    </AuthContext.Provider>
  );
};

const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Components
const ConfidenceBar = ({ confidence, outcome }) => {
  const getColor = () => {
    if (confidence >= 0.7) return 'bg-green-500';
    if (confidence >= 0.5) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getTextColor = () => {
    if (confidence >= 0.7) return 'text-green-400';
    if (confidence >= 0.5) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className={`text-sm font-medium ${getTextColor()}`}>
          {outcome.toUpperCase()} ({(confidence * 100).toFixed(1)}%)
        </span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2">
        <div
          className={`${getColor()} h-2 rounded-full transition-all duration-500`}
          style={{ width: `${confidence * 100}%` }}
        />
      </div>
    </div>
  );
};

const MatchCard = ({ match, prediction }) => {
  const getOddsColor = (odds, minOdds) => {
    return odds === minOdds ? 'text-green-400 font-bold' : 'text-gray-300';
  };

  const minOdds = Math.min(match.home_odds, match.draw_odds, match.away_odds);

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-gray-600 transition-colors">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white mb-1">{match.match}</h3>
          <p className="text-sm text-gray-400">{match.league}</p>
        </div>
        <div className="text-xs text-gray-500">
          {new Date(match.match_date).toLocaleDateString()}
        </div>
      </div>

      {/* Odds */}
      <div className="grid grid-cols-3 gap-4 mb-4 p-3 bg-gray-900 rounded">
        <div className="text-center">
          <div className="text-xs text-gray-400 mb-1">Home</div>
          <div className={`text-lg ${getOddsColor(match.home_odds, minOdds)}`}>
            {match.home_odds}
          </div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-400 mb-1">Draw</div>
          <div className={`text-lg ${getOddsColor(match.draw_odds, minOdds)}`}>
            {match.draw_odds}
          </div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-400 mb-1">Away</div>
          <div className={`text-lg ${getOddsColor(match.away_odds, minOdds)}`}>
            {match.away_odds}
          </div>
        </div>
      </div>

      {/* AI Prediction */}
      {prediction && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-4 h-4 text-blue-400" />
            <span className="text-sm font-medium text-blue-400">AI Prediction</span>
          </div>
          
          <ConfidenceBar 
            confidence={prediction.confidence} 
            outcome={prediction.predicted_outcome} 
          />
          
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="text-center">
              <div className="text-gray-400">Home</div>
              <div className="text-white">{(prediction.home_probability * 100).toFixed(1)}%</div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">Draw</div>
              <div className="text-white">{(prediction.draw_probability * 100).toFixed(1)}%</div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">Away</div>
              <div className="text-white">{(prediction.away_probability * 100).toFixed(1)}%</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const Dashboard = () => {
  const [matches, setMatches] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const { user, isAdmin } = useAuth();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      
      // Get odds and predictions
      const [oddsResponse, predictionsResponse] = await Promise.all([
        axios.get(`${API}/odds`),
        axios.get(`${API}/predictions`)
      ]);
      
      setMatches(oddsResponse.data);
      setPredictions(predictionsResponse.data);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getPredictionForMatch = (matchId) => {
    return predictions.find(p => p.match_id === matchId);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 text-blue-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading predictions...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center gap-3">
              <TrendingUp className="w-8 h-8 text-blue-400" />
              <h1 className="text-2xl font-bold text-white">PredictBet AI</h1>
            </div>
            
            <div className="flex items-center gap-4">
              {user && (
                <div className="flex items-center gap-3 text-sm text-gray-300">
                  <Mail className="w-4 h-4" />
                  <span>{user.email}</span>
                  {isAdmin() && (
                    <div className="flex items-center gap-1 px-2 py-1 bg-blue-900 rounded text-blue-300">
                      <Shield className="w-3 h-3" />
                      <span className="text-xs">Admin</span>
                    </div>
                  )}
                </div>
              )}
              
              <div className="flex gap-2">
                {isAdmin() && (
                  <Link 
                    to="/admin" 
                    className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition-colors"
                  >
                    Admin Panel
                  </Link>
                )}
                <button 
                  onClick={() => window.location.reload()}
                  className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm transition-colors"
                  title="Refresh"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-white mb-2">Match Predictions</h2>
          <p className="text-gray-400">AI-powered betting predictions with confidence indicators</p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {matches.map(match => (
            <MatchCard 
              key={match.id} 
              match={match} 
              prediction={getPredictionForMatch(match.id)} 
            />
          ))}
        </div>

        {matches.length === 0 && (
          <div className="text-center py-12">
            <Activity className="w-12 h-12 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400">No matches available</p>
          </div>
        )}
      </main>
    </div>
  );
};

const AdminPanel = () => {
  const [stats, setStats] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [file, setFile] = useState(null);
  const [training, setTraining] = useState(false);
  const [message, setMessage] = useState('');
  const { user, isAdmin, signOut } = useAuth();

  useEffect(() => {
    if (isAdmin()) {
      loadAdminData();
    }
  }, []);

  const loadAdminData = async () => {
    try {
      const token = (await supabase.auth.getSession()).data.session?.access_token;
      const headers = token ? { Authorization: `Bearer ${token}` } : {};

      const [statsResponse, statusResponse] = await Promise.all([
        axios.get(`${API}/admin/stats`, { headers }),
        axios.get(`${API}/model/status`, { headers })
      ]);

      setStats(statsResponse.data);
      setModelStatus(statusResponse.data);
    } catch (error) {
      console.error('Error loading admin data:', error);
      setMessage('Error loading admin data');
    }
  };

  const handleFileUpload = async () => {
    if (!file) {
      setMessage('Please select a CSV file');
      return;
    }

    try {
      setTraining(true);
      setMessage('Training model...');

      const token = (await supabase.auth.getSession()).data.session?.access_token;
      const formData = new FormData();
      formData.append('file', file);

      await axios.post(`${API}/train`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          Authorization: `Bearer ${token}`
        }
      });

      setMessage('Model trained successfully!');
      setFile(null);
      loadAdminData();
    } catch (error) {
      setMessage(`Training failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setTraining(false);
    }
  };

  if (!isAdmin()) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <Shield className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-2">Access Denied</h2>
          <p className="text-gray-400 mb-6">You need admin privileges to access this panel</p>
          <Link to="/" className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg">
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center gap-3">
              <Shield className="w-8 h-8 text-blue-400" />
              <h1 className="text-2xl font-bold text-white">Admin Panel</h1>
            </div>
            
            <div className="flex items-center gap-4">
              <Link 
                to="/" 
                className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm"
              >
                Back to Dashboard
              </Link>
              <button 
                onClick={signOut}
                className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm flex items-center gap-2"
              >
                <LogOut className="w-4 h-4" />
                Sign Out
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Admin Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <div className="flex items-center gap-3 mb-2">
              <Activity className="w-5 h-5 text-blue-400" />
              <h3 className="text-sm font-medium text-gray-400">Total Matches</h3>
            </div>
            <p className="text-2xl font-bold text-white">{stats?.total_matches || 0}</p>
          </div>
          
          <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <div className="flex items-center gap-3 mb-2">
              <Target className="w-5 h-5 text-green-400" />
              <h3 className="text-sm font-medium text-gray-400">Predictions</h3>
            </div>
            <p className="text-2xl font-bold text-white">{stats?.total_predictions || 0}</p>
          </div>
          
          <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <div className="flex items-center gap-3 mb-2">
              <BarChart3 className="w-5 h-5 text-yellow-400" />
              <h3 className="text-sm font-medium text-gray-400">Model Accuracy</h3>
            </div>
            <p className="text-2xl font-bold text-white">
              {stats?.model_accuracy ? `${(stats.model_accuracy * 100).toFixed(1)}%` : 'N/A'}
            </p>
          </div>
          
          <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <div className="flex items-center gap-3 mb-2">
              <Calendar className="w-5 h-5 text-purple-400" />
              <h3 className="text-sm font-medium text-gray-400">Last Retrain</h3>
            </div>
            <p className="text-sm font-bold text-white">
              {stats?.last_retrain 
                ? new Date(stats.last_retrain).toLocaleDateString()
                : 'Never'
              }
            </p>
          </div>
        </div>

        {/* Model Training */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-8">
          <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Model Training
          </h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Upload Training Data (CSV)
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setFile(e.target.files[0])}
                className="block w-full text-sm text-gray-300 bg-gray-700 border border-gray-600 rounded-lg cursor-pointer focus:outline-none"
              />
              <p className="mt-1 text-xs text-gray-400">
                CSV should have columns: home_odds, draw_odds, away_odds, result
              </p>
            </div>
            
            <button
              onClick={handleFileUpload}
              disabled={training || !file}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg flex items-center gap-2 transition-colors"
            >
              {training ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Training...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4" />
                  Train Model
                </>
              )}
            </button>
          </div>
        </div>

        {/* Model Status */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Model Status
          </h2>
          
          {modelStatus && (
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">Status:</span>
                <span className={`font-medium ${
                  modelStatus.model_exists ? 'text-green-400' : 'text-red-400'
                }`}>
                  {modelStatus.status}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-400">Last Trained:</span>
                <span className="text-white">
                  {modelStatus.last_trained 
                    ? new Date(modelStatus.last_trained).toLocaleString()
                    : 'Never'
                  }
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-400">Accuracy:</span>
                <span className="text-white">
                  {modelStatus.accuracy 
                    ? `${(modelStatus.accuracy * 100).toFixed(2)}%`
                    : 'N/A'
                  }
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-400">Total Predictions:</span>
                <span className="text-white">{modelStatus.total_predictions}</span>
              </div>
            </div>
          )}
        </div>

        {/* Messages */}
        {message && (
          <div className={`mt-4 p-4 rounded-lg ${
            message.includes('Error') || message.includes('failed') 
              ? 'bg-red-900 border border-red-700 text-red-300'
              : 'bg-green-900 border border-green-700 text-green-300'
          }`}>
            {message}
          </div>
        )}
      </main>
    </div>
  );
};

const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [showDemo, setShowDemo] = useState(false);
  const { signIn } = useAuth();

  const handleSignIn = async (e) => {
    e.preventDefault();
    if (!email) return;

    setLoading(true);
    setMessage('');

    const { error } = await signIn(email);

    if (error) {
      setMessage(`Error: ${error.message}`);
    } else {
      setMessage('Check your email for the login link!');
    }

    setLoading(false);
  };

  const handleDemoAccess = () => {
    // For demo purposes, navigate directly to dashboard
    window.location.href = '/dashboard';
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center">
      <div className="max-w-md w-full space-y-8 p-8">
        <div className="text-center">
          <TrendingUp className="w-16 h-16 text-blue-400 mx-auto mb-4" />
          <h2 className="text-3xl font-bold text-white">PredictBet AI</h2>
          <p className="mt-2 text-sm text-gray-400">
            AI-powered sports betting predictions
          </p>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          {!showDemo ? (
            <>
              <form onSubmit={handleSignIn} className="space-y-4">
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-300">
                    Email address
                  </label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="mt-1 block w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Enter your email"
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 transition-colors"
                >
                  {loading ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    'Sign in with email'
                  )}
                </button>
              </form>

              <div className="mt-4">
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-gray-600" />
                  </div>
                  <div className="relative flex justify-center text-sm">
                    <span className="px-2 bg-gray-800 text-gray-400">Or</span>
                  </div>
                </div>

                <button
                  onClick={() => setShowDemo(true)}
                  className="mt-4 w-full flex justify-center py-2 px-4 border border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-300 bg-gray-700 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 transition-colors"
                >
                  <Eye className="w-4 h-4 mr-2" />
                  View Demo
                </button>
              </div>
            </>
          ) : (
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-white">Demo Access</h3>
              <p className="text-sm text-gray-400">
                Explore PredictBet AI without signing up. Full features available with email authentication.
              </p>
              
              <div className="space-y-2">
                <button
                  onClick={handleDemoAccess}
                  className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition-colors"
                >
                  <Eye className="w-4 h-4 mr-2" />
                  Enter as Public User
                </button>
                
                <button
                  onClick={() => window.location.href = '/admin'}
                  className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
                >
                  <Shield className="w-4 h-4 mr-2" />
                  View Admin Panel
                </button>
              </div>

              <button
                onClick={() => setShowDemo(false)}
                className="w-full flex justify-center py-2 px-4 border border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-300 bg-gray-700 hover:bg-gray-600 transition-colors"
              >
                <EyeOff className="w-4 h-4 mr-2" />
                Back to Login
              </button>
            </div>
          )}

          {message && (
            <div className={`mt-4 p-3 rounded-md text-sm ${
              message.includes('Error') 
                ? 'bg-red-900 border border-red-700 text-red-300'
                : 'bg-green-900 border border-green-700 text-green-300'
            }`}>
              {message}
            </div>
          )}
        </div>

        <div className="text-center">
          <p className="text-xs text-gray-500">
            Admin demo: Use any email ending with @admin.com
          </p>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/admin" element={<AdminPanel />} />
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </div>
  );
}

export default App;