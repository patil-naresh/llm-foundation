# LLM Fundamentals

This repository is organized as two independent applications:

- [`backend/`](./backend/README.md) contains the FastAPI service, API tests, and
  standalone LLM proof-of-concept scripts.
- [`frontend/`](./frontend/README.md) contains the TanStack/React web app.

## Run locally

Start the backend in one terminal:

```powershell
cd backend
python -m pip install -r requirements-dev.txt
python app.py
```

In another terminal, start the frontend:

```powershell
cd frontend
npm ci
npm run dev
```

Configure backend credentials in `backend/.env` using `backend/.env.example`.
Frontend configuration such as `VITE_API_BASE_URL` belongs in
`frontend/.env.local`; never put provider API keys in frontend environment
variables.
