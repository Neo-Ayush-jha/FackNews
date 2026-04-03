# Fake News Detector - Run Steps (Short)

## 1) Open terminal in project root
Open terminal in the main folder (where `backend/` exists).

## 2) Go to backend folder
```powershell
cd backend
```

## 3) Set up virtual environment
If `env/` already exists in the project root, activate it:
```powershell
..\env\Scripts\Activate.ps1
```

If `env/` does not exist, create and activate it:
```powershell
cd ..
python -m venv env
.\env\Scripts\Activate.ps
cd backend
```

## 4) Install dependencies
```powershell
pip install -r requirements.txt
```

## 5) Run database migrations
```powershell
python manage.py migrate
```

## 6) Start Django server
```powershell
python manage.py runserver
```

## 7) Open in browser
- App/API root: `http://127.0.0.1:8000/`
- Prediction API (POST): `http://127.0.0.1:8000/api/predict/`

## Optional: Run model training
Use this only if you want to retrain the model:
```powershell
python train_model.py
```

## Optional: Quick API test
```powershell
curl -X POST http://127.0.0.1:8000/api/predict/ -H "Content-Type: application/json" -d '{"text":"Breaking news sample text"}'
```
