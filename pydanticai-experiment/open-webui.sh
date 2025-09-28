WEBUI_AUTH=False \
ENABLE_PERSISTENT_CONFIG=False \
DATA_DIR=$(pwd)/.open-webui-data \
ENABLE_FOLLOW_UP_GENERATION=False \
ENABLE_OPENAI_API=True \
OPENAI_API_BASE_URL=http://localhost:8000/v1 \
OPENAI_API_KEY=dummy-api-key \
uv tool run --python 3.12 open-webui serve \
    --host 127.0.0.1 \
    --port 8080
