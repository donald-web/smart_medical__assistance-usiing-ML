# Smart Medical Assistant Final Full Code

## Setup

1. Create PostgreSQL database:
```sql
CREATE DATABASE smart_medical_assistant_db;
```

2. Rename `.env.example` to `.env` and update password:
```env
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/smart_medical_assistant_db
```

3. Create and activate venv:
```powershell
python -m venv venv
venv\Scripts\activate
```

4. Install requirements:
```powershell
pip install -r requirements.txt
```

5. Train model:
```powershell
python train_model.py
```

6. Run app:
```powershell
python app.py
```

7. Open:
```text
http://127.0.0.1:5000
```

If database table error appears, run in pgAdmin:
```sql
DROP TABLE IF EXISTS prediction_history CASCADE;
DROP TABLE IF EXISTS appointment CASCADE;
```
Then run `python app.py` again.

Voice input works best in Google Chrome.
