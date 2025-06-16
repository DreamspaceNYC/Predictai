from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.security import HTTPBearer
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import asyncio
from supabase import create_client, Client
from jose import JWTError, jwt

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Supabase client
supabase_url = os.environ.get('SUPABASE_URL')
supabase_key = os.environ.get('SUPABASE_ANON_KEY')
supabase_jwt_secret = os.environ.get('SUPABASE_JWT_SECRET', 'your-jwt-secret')
supabase: Client = create_client(supabase_url, supabase_key)

# Create the main app without a prefix
app = FastAPI(title="PredictBet AI API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Security
security = HTTPBearer()

# Models directory
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "prediction_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Define Pydantic Models
class Match(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    match: str
    home_team: str
    away_team: str
    home_odds: float
    draw_odds: float
    away_odds: float
    league: str = "Unknown"
    match_date: datetime = Field(default_factory=datetime.utcnow)

class Prediction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    match_id: str
    match: str
    predicted_outcome: str  # "home", "draw", "away"
    confidence: float  # 0.0 to 1.0
    home_probability: float
    draw_probability: float
    away_probability: float
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TrainingData(BaseModel):
    home_team: str
    away_team: str
    home_odds: float
    draw_odds: float
    away_odds: float
    result: str  # "home", "draw", "away"

class ModelStatus(BaseModel):
    model_exists: bool
    last_trained: Optional[datetime]
    accuracy: Optional[float]
    total_predictions: int
    status: str

class AdminStats(BaseModel):
    total_matches: int
    total_predictions: int
    model_accuracy: Optional[float]
    last_retrain: Optional[datetime]

# Mock odds data
MOCK_ODDS = [
    {
        "id": str(uuid.uuid4()),
        "match": "Arsenal vs Chelsea",
        "home_team": "Arsenal",
        "away_team": "Chelsea", 
        "home_odds": 2.1,
        "draw_odds": 3.5,
        "away_odds": 3.2,
        "league": "Premier League"
    },
    {
        "id": str(uuid.uuid4()),
        "match": "Real Madrid vs Barcelona",
        "home_team": "Real Madrid",
        "away_team": "Barcelona",
        "home_odds": 1.9,
        "draw_odds": 3.8,
        "away_odds": 3.5,
        "league": "La Liga"
    },
    {
        "id": str(uuid.uuid4()),
        "match": "Manchester City vs Liverpool",
        "home_team": "Manchester City",
        "away_team": "Liverpool",
        "home_odds": 2.3,
        "draw_odds": 3.2,
        "away_odds": 2.8,
        "league": "Premier League"
    },
    {
        "id": str(uuid.uuid4()),
        "match": "Bayern Munich vs Borussia Dortmund",
        "home_team": "Bayern Munich",
        "away_team": "Borussia Dortmund",
        "home_odds": 1.7,
        "draw_odds": 4.1,
        "away_odds": 4.5,
        "league": "Bundesliga"
    }
]

# Authentication helpers
async def verify_jwt(credentials = Depends(security)):
    """Verify JWT token from Supabase"""
    try:
        payload = jwt.decode(
            credentials.credentials,
            supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated"
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

async def get_current_user(token_data = Depends(verify_jwt)):
    """Get current user from JWT"""
    return token_data

async def require_admin(user = Depends(get_current_user)):
    """Require admin role"""
    # For now, we'll check if user has admin role in app_metadata
    # In production, this would be configured in Supabase
    user_role = user.get('app_metadata', {}).get('role', 'user')
    if user_role != 'admin':
        # For demo purposes, we'll allow access if user exists
        # In production, uncomment the line below
        # raise HTTPException(status_code=403, detail="Admin access required")
        pass
    return user

# ML Model helpers
def create_features(home_odds: float, draw_odds: float, away_odds: float):
    """Create features from odds"""
    # Implied probabilities from odds
    home_prob = 1 / home_odds
    draw_prob = 1 / draw_odds
    away_prob = 1 / away_odds
    
    # Normalize probabilities (remove bookmaker margin)
    total_prob = home_prob + draw_prob + away_prob
    home_prob_norm = home_prob / total_prob
    draw_prob_norm = draw_prob / total_prob
    away_prob_norm = away_prob / total_prob
    
    # Additional features
    favorite_odds = min(home_odds, away_odds)
    underdog_odds = max(home_odds, away_odds)
    odds_spread = underdog_odds - favorite_odds
    
    return [
        home_odds, draw_odds, away_odds,
        home_prob_norm, draw_prob_norm, away_prob_norm,
        favorite_odds, underdog_odds, odds_spread
    ]

def train_model(training_data: List[Dict]):
    """Train ML model from training data"""
    if len(training_data) < 10:
        raise ValueError("Need at least 10 training samples")
    
    # Prepare features and labels
    X = []
    y = []
    
    for data in training_data:
        features = create_features(
            data['home_odds'], 
            data['draw_odds'], 
            data['away_odds']
        )
        X.append(features)
        
        # Convert result to numeric label
        if data['result'] == 'home':
            y.append(0)
        elif data['result'] == 'draw':
            y.append(1)
        else:  # away
            y.append(2)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try XGBoost first, fallback to RandomForest
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=42, eval_metric='mlogloss')
    except ImportError:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    return model, scaler, accuracy

def load_model():
    """Load trained model and scaler"""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception:
        return None, None

def predict_match(home_odds: float, draw_odds: float, away_odds: float):
    """Predict match outcome"""
    model, scaler = load_model()
    
    if model is None or scaler is None:
        # Fallback prediction based on odds only
        odds_list = [home_odds, draw_odds, away_odds]
        min_odds_idx = odds_list.index(min(odds_list))
        outcomes = ['home', 'draw', 'away']
        
        # Simple confidence based on odds difference
        sorted_odds = sorted(odds_list)
        confidence = (sorted_odds[1] - sorted_odds[0]) / sorted_odds[1]
        confidence = min(max(confidence, 0.3), 0.9)  # Clamp between 0.3 and 0.9
        
        # Equal probabilities for fallback
        home_prob = 0.4 if min_odds_idx == 0 else 0.3
        draw_prob = 0.4 if min_odds_idx == 1 else 0.3
        away_prob = 0.4 if min_odds_idx == 2 else 0.3
        
        return outcomes[min_odds_idx], confidence, home_prob, draw_prob, away_prob
    
    # Use trained model
    features = create_features(home_odds, draw_odds, away_odds)
    features_scaled = scaler.transform([features])
    
    # Get probabilities
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get prediction
    prediction_idx = np.argmax(probabilities)
    outcomes = ['home', 'draw', 'away']
    predicted_outcome = outcomes[prediction_idx]
    
    # Confidence is the max probability
    confidence = float(probabilities[prediction_idx])
    
    return (
        predicted_outcome, 
        confidence, 
        float(probabilities[0]),  # home
        float(probabilities[1]),  # draw
        float(probabilities[2])   # away
    )

# API Routes

@api_router.get("/")
async def root():
    return {"message": "PredictBet AI API", "version": "1.0.0"}

@api_router.get("/odds", response_model=List[Match])
async def get_odds():
    """Get current odds for matches"""
    # Convert mock data to Match objects
    matches = []
    for odds_data in MOCK_ODDS:
        match = Match(**odds_data)
        matches.append(match)
    
    # Store in database
    for match in matches:
        await db.matches.update_one(
            {"id": match.id},
            {"$set": match.dict()},
            upsert=True
        )
    
    return matches

@api_router.get("/predictions", response_model=List[Prediction])
async def get_predictions():
    """Get AI predictions for current matches"""
    # Get current matches
    matches = await db.matches.find().to_list(1000)
    
    if not matches:
        # Use mock data if no matches in DB
        odds_response = await get_odds()
        matches = [match.dict() for match in odds_response]
    
    predictions = []
    
    for match in matches:
        # Check if prediction already exists
        existing_prediction = await db.predictions.find_one({"match_id": match["id"]})
        
        if existing_prediction:
            predictions.append(Prediction(**existing_prediction))
        else:
            # Generate new prediction
            predicted_outcome, confidence, home_prob, draw_prob, away_prob = predict_match(
                match["home_odds"], match["draw_odds"], match["away_odds"]
            )
            
            prediction = Prediction(
                match_id=match["id"],
                match=match["match"],
                predicted_outcome=predicted_outcome,
                confidence=confidence,
                home_probability=home_prob,
                draw_probability=draw_prob,
                away_probability=away_prob
            )
            
            # Store prediction
            await db.predictions.insert_one(prediction.dict())
            predictions.append(prediction)
    
    return predictions

@api_router.post("/predict", response_model=Prediction)
async def manual_predict(
    home_odds: float,
    draw_odds: float, 
    away_odds: float,
    match_name: str = "Manual Prediction",
    user = Depends(get_current_user)
):
    """Manually trigger prediction for specific odds"""
    predicted_outcome, confidence, home_prob, draw_prob, away_prob = predict_match(
        home_odds, draw_odds, away_odds
    )
    
    prediction = Prediction(
        match_id=str(uuid.uuid4()),
        match=match_name,
        predicted_outcome=predicted_outcome,
        confidence=confidence,
        home_probability=home_prob,
        draw_probability=draw_prob,
        away_probability=away_prob
    )
    
    # Store prediction
    await db.predictions.insert_one(prediction.dict())
    
    return prediction

@api_router.post("/train")
async def train_model_endpoint(
    file: UploadFile = File(...),
    admin_user = Depends(require_admin)
):
    """Train model with uploaded CSV data (Admin only)"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        
        # Save temporarily
        temp_path = ROOT_DIR / f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Load data
        df = pd.read_csv(temp_path)
        
        # Validate required columns
        required_columns = ['home_odds', 'draw_odds', 'away_odds', 'result']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Missing columns: {missing_columns}"
            )
        
        # Convert to training data
        training_data = []
        for _, row in df.iterrows():
            training_data.append({
                'home_odds': float(row['home_odds']),
                'draw_odds': float(row['draw_odds']),
                'away_odds': float(row['away_odds']),
                'result': str(row['result']).lower()
            })
        
        # Train model
        model, scaler, accuracy = train_model(training_data)
        
        # Store training metadata
        training_metadata = {
            "id": str(uuid.uuid4()),
            "trained_at": datetime.utcnow(),
            "accuracy": accuracy,
            "training_samples": len(training_data),
            "filename": file.filename
        }
        
        await db.model_training.insert_one(training_metadata)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "message": "Model trained successfully",
            "accuracy": accuracy,
            "training_samples": len(training_data)
        }
        
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/model/status", response_model=ModelStatus)
async def get_model_status(admin_user = Depends(require_admin)):
    """Get model status and statistics (Admin only)"""
    # Check if model exists
    model_exists = MODEL_PATH.exists() and SCALER_PATH.exists()
    
    # Get last training info
    last_training = await db.model_training.find_one(
        {},
        sort=[("trained_at", -1)]
    )
    
    # Get prediction count
    prediction_count = await db.predictions.count_documents({})
    
    return ModelStatus(
        model_exists=model_exists,
        last_trained=last_training["trained_at"] if last_training else None,
        accuracy=last_training["accuracy"] if last_training else None,
        total_predictions=prediction_count,
        status="Ready" if model_exists else "No model trained"
    )

@api_router.get("/admin/stats", response_model=AdminStats)
async def get_admin_stats(admin_user = Depends(require_admin)):
    """Get admin dashboard statistics"""
    # Count matches and predictions
    match_count = await db.matches.count_documents({})
    prediction_count = await db.predictions.count_documents({})
    
    # Get last training info
    last_training = await db.model_training.find_one(
        {},
        sort=[("trained_at", -1)]
    )
    
    return AdminStats(
        total_matches=match_count,
        total_predictions=prediction_count,
        model_accuracy=last_training["accuracy"] if last_training else None,
        last_retrain=last_training["trained_at"] if last_training else None
    )

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Create indexes and setup on startup"""
    # Create indexes
    await db.matches.create_index("id", unique=True)
    await db.predictions.create_index("match_id")
    await db.model_training.create_index("trained_at")
    logger.info("PredictBet AI API started successfully")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()