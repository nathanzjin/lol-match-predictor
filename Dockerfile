# Container image for the LoL match predictor web app.
#
# The match data (~290 MB) and the trained model are NOT in git, so the build
# fetches the data and trains the model, baking both into the image. The running
# container then serves instantly (the data is loaded into memory at startup and
# the model is read from disk). Rebuild to refresh the data / extend the season
# track record.
FROM python:3.11-slim

# libgomp1 is the OpenMP runtime xgboost needs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bootstrap: download Oracle's Elixir data, then train + save the model.
RUN python download_data.py && python train_v3.py

# Hosts inject $PORT; default to 8000 locally.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
