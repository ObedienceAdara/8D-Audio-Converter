web: bash -c "apt-get update && apt-get install -y ffmpeg && exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120"
