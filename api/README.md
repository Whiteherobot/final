# API Transaccional

API mínima para consultas transaccionales con Neo4j y Gemini.

## Ejecutar

1. Crear entorno e instalar dependencias:
   pip install -r api/requirements.txt

2. Levantar el servidor:
   uvicorn api.main:app --reload

## Endpoints

- GET /health
- POST /query

Body:
{
  "query": "¿Cuánto cuesta el iPhone 14?"
}
