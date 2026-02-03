from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import google.generativeai as genai
import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(title="Transaccional API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Servir archivos estáticos del frontend
front_path = BASE_DIR / "front"
if front_path.exists():
    app.mount("/static", StaticFiles(directory=str(front_path)), name="static")

# Cargar modelo de embeddings con fallback
embedder = None
try:
    print("Cargando modelo de embeddings...")
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("[OK] Modelo cargado exitosamente")
except Exception as e:
    print(f"[WARN] No se pudo cargar modelo de embeddings: {e}")
    print("  El API funcionará con vectores simulados")
    embedder = None

# Intentar conectar a Neo4j, pero continuar si no está disponible
driver = None
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
except Exception as e:
    print(f"[WARN] Advertencia: No se pudo conectar a Neo4j ({e})")
    print("  El API funcionará en modo simulado con datos de ejemplo")
    driver = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    function: str
    similitud: float = 0.0
    results: list
    answer: str
    action: Optional[str] = None
    cart_items: Optional[List[dict]] = None

class PedidoItem(BaseModel):
    producto: str
    cantidad: int
    precio: float

class PedidoRequest(BaseModel):
    items: List[PedidoItem]

class PedidoResponse(BaseModel):
    id: str
    timestamp: str
    items: List[PedidoItem]
    total: float
    estado: str = "completado"

FUNCIONES_SISTEMA = [
    {
        "id": 1,
        "nombre_funcion": "buscar_producto",
        "descripcion": "Busca productos por nombre o categoria. Retorna precio y vendedor.",
        "query_examples": ["Cuanto cuesta", "precio", "producto", "categoría"],
    },
    {
        "id": 2,
        "nombre_funcion": "buscar_vendedor",
        "descripcion": "Busca vendedores por ciudad o desempeño.",
        "query_examples": ["vendedor", "quien vende", "ciudad"],
    },
    {
        "id": 3,
        "nombre_funcion": "verificar_stock",
        "descripcion": "Consulta disponibilidad de productos.",
        "query_examples": ["stock", "disponible", "hay"],
    },
    {
        "id": 4,
        "nombre_funcion": "consultar_clientes",
        "descripcion": "Consulta cantidad de clientes y distribución básica.",
        "query_examples": ["clientes", "cuantos clientes", "usuarios"],
    },
]

# Almacenar pedidos en memoria (en producción usar base de datos)
pedidos_almacenados: List[dict] = []
PEDIDOS_FILE = BASE_DIR / "data" / "pedidos.json"

# Crear directorio data si no existe
(BASE_DIR / "data").mkdir(exist_ok=True)

# Cargar pedidos previos
def cargar_pedidos():
    global pedidos_almacenados
    if PEDIDOS_FILE.exists():
        try:
            with open(PEDIDOS_FILE, "r", encoding="utf-8") as f:
                pedidos_almacenados = json.load(f)
        except:
            pedidos_almacenados = []
    else:
        pedidos_almacenados = []

def guardar_pedidos():
    """Guarda pedidos a archivo JSON"""
    try:
        PEDIDOS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PEDIDOS_FILE, "w", encoding="utf-8") as f:
            json.dump(pedidos_almacenados, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error guardando pedidos: {e}")

# Cargar pedidos al iniciar
cargar_pedidos()

# Cargar datasets reales
df_productos = None
df_vendedores = None

try:
    datos_path = BASE_DIR / "data"
    df_productos = pd.read_csv(datos_path / "dataset_transaccional.csv", encoding='utf-8')
    df_vendedores = pd.read_csv(datos_path / "dataset_vendedores.csv", encoding='utf-8')
    print("[OK] Datasets cargados exitosamente")
    print(f"  - Productos: {len(df_productos)} registros")
    print(f"  - Vendedores: {len(df_vendedores)} registros")
except Exception as e:
    print(f"[WARN] Error cargando datasets: {e}")
    print("  El API funcionará con datos simulados")
    df_productos = None
    df_vendedores = None

# Generar embeddings solo si el modelo está disponible
if embedder:
    for funcion in FUNCIONES_SISTEMA:
        texto = f"{funcion['nombre_funcion']}: {funcion['descripcion']}. Ejemplos: {', '.join(funcion['query_examples'])}"
        funcion["embedding"] = embedder.encode(texto)
else:
    # Usar embeddings simulados
    for i, funcion in enumerate(FUNCIONES_SISTEMA):
        funcion["embedding"] = [0.0] * 384  # Simulado: 384 dimensiones


def seleccionar_funcion(query: str):
    """Retorna tupla (nombre_funcion, similitud)"""
    q = query.lower()

    product_keywords = [
        "producto", "productos", "categoria", "categoría", "precio", "cuesta", "costo", "vale", "cuanto",
        "laptop", "laptops", "notebook", "audifono", "audífono", "audifonos", "audífonos",
        "iphone", "smartphone", "celular", "computadora", "macbook", "dell", "samsung", "sony"
    ]
    product_specific = [
        "laptop", "laptops", "notebook", "audifono", "audífono", "audifonos", "audífonos",
        "iphone", "smartphone", "celular", "computadora", "macbook", "dell", "samsung", "sony"
    ]
    vendor_keywords = ["vendedor", "vendedores", "vende", "venden", "quien vende", "quién vende", "seller", "tienda"]
    stock_keywords = ["stock", "disponible", "disponibilidad", "existencia"]
    customer_keywords = ["cliente", "clientes", "usuarios", "compradores", "cuantos clientes", "cuántos clientes"]

    has_product = any(k in q for k in product_keywords)
    has_specific_product = any(k in q for k in product_specific)
    has_vendor = any(k in q for k in vendor_keywords)
    has_stock = any(k in q for k in stock_keywords)
    has_customer = any(k in q for k in customer_keywords)

    if has_customer:
        return ("consultar_clientes", 1.0)
    # Si hay stock y no hay producto específico, ir a stock
    if has_stock and not has_specific_product and not has_vendor:
        return ("verificar_stock", 1.0)
    # Priorizar producto si hay intención de producto específica
    if has_product and not has_vendor:
        return ("buscar_producto", 1.0)
    if has_vendor and not has_product:
        return ("buscar_vendedor", 1.0)
    if has_stock:
        return ("verificar_stock", 1.0)

    # Si embedder está disponible, usar similitud; si no, usar fallback
    if embedder:
        emb = embedder.encode(query)
        scores = []
        for f in FUNCIONES_SISTEMA:
            sim = float(cosine_similarity([emb], [f["embedding"]])[0][0])
            scores.append((f["nombre_funcion"], sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return (scores[0][0], scores[0][1])
    else:
        # Fallback: retornar función por defecto con baja similitud
        return ("buscar_producto", 0.5)


def run_query(cypher: str, params=None):
    """Ejecuta query en Neo4j si está disponible, si no retorna datos simulados"""
    if driver:
        try:
            with driver.session() as session:
                result = session.run(cypher, params or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"Error ejecutando query: {e}")
            return get_mock_data(cypher)
    else:
        return get_mock_data(cypher)


def extraer_categoria(query: str):
    q = (query or "").lower()
    
    # Si tenemos datos, buscar categorías disponibles
    if df_productos is not None:
        categorias_disponibles = df_productos['categoria'].unique()
        for cat in categorias_disponibles:
            if cat and cat.lower() in q:
                return cat
    
    # Fallback a búsqueda por palabras clave
    categorias_fallback = {
        "audio": ["audio", "audifono", "audífono", "audifonos", "audífonos", "parlante", "speaker"],
        "laptops": ["laptop", "laptops", "notebook", "computadora", "portatil", "portátil"],
        "smartphones": ["smartphone", "celular", "móvil", "movil", "iphone", "samsung"],
    }
    for cat, keys in categorias_fallback.items():
        if any(k in q for k in keys):
            return cat
    return None


def _normalizar_tokens(query: str):
    raw_tokens = [t for t in re.split(r"[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ]+", query.lower()) if len(t) >= 2]
    stopwords = {
        "que", "hay", "disponible", "disponibles", "precio", "cuanto", "cuánto",
        "productos", "producto", "de", "el", "la", "los", "las", "quiero",
        "comprar", "agregar", "pedido", "orden", "un", "una", "unos", "unas"
    }
    tokens = [t for t in raw_tokens if t not in stopwords]
    normalized_tokens = []
    for t in tokens:
        normalized_tokens.append(t)
        if t.endswith("s") and len(t) > 3:
            normalized_tokens.append(t[:-1])
    return list(dict.fromkeys(normalized_tokens))


def parse_order_intent(query: str):
    """Detecta intención de agregar al carrito desde texto libre."""
    q = (query or "").lower()
    if not any(k in q for k in ["quiero", "comprar", "agregar", "pedido", "orden"]):
        return None

    if df_productos is None:
        return None

    # Cantidad (número o palabra)
    cantidad = 1
    m = re.search(r"\b(\d+)\b", q)
    if m:
        try:
            cantidad = max(1, int(m.group(1)))
        except Exception:
            cantidad = 1
    else:
        num_words = {
            "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
            "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10
        }
        for w, n in num_words.items():
            if w in q:
                cantidad = n
                break

    tokens = _normalizar_tokens(query)
    if not tokens:
        return None

    # Buscar el mejor match en nombre/descripcion
    best_row = None
    best_score = 0
    for _, row in df_productos.iterrows():
        nombre = str(row.get("nombre", "")).lower()
        descripcion = str(row.get("descripcion", "")).lower()
        score = 0
        for tok in tokens:
            if tok in nombre or tok in descripcion:
                score += 1
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None or best_score == 0:
        return None

    return [{
        "producto": best_row.get("nombre"),
        "cantidad": cantidad,
        "precio": float(best_row.get("precio"))
    }]


def get_mock_data(query_type: str = "default"):
    """Retorna datos del dataset real o simula si no está disponible"""
    if query_type == "producto" and df_productos is not None:
        return df_productos.head(10).to_dict('records')
    elif query_type == "vendedor" and df_vendedores is not None:
        return df_vendedores.head(10).to_dict('records')
    elif query_type == "stock" and df_productos is not None:
        datos = df_productos[['nombre', 'categoria', 'precio', 'stock']].head(10).copy()
        datos.columns = ['producto', 'categoria', 'precio', 'disponible']
        return datos.to_dict('records')
    elif query_type == "clientes":
        if df_productos is not None:
            return [{"total_clientes": len(df_productos)}]
        return [{"total_clientes": 5234}]
    else:
        # Fallback a datos simulados
        return [
            {"categoria": "Electronica", "cantidad_productos": 10, "precio_minimo": 299.99, "precio_maximo": 1299.99},
        ]


def buscar_producto(query: str):
    categoria = extraer_categoria(query)
    
    # Usar dataset real si está disponible
    if df_productos is not None:
        data = df_productos.copy()

        # Si la categoría detectada no existe en el dataset, ignorarla
        if categoria:
            categorias_dataset = set(df_productos['categoria'].astype(str).str.lower().unique())
            if str(categoria).lower() not in categorias_dataset:
                categoria = None

        # Filtrar por categoría si se detectó
        if categoria:
            data = data[data['categoria'].str.lower() == str(categoria).lower()]

        # Tokenizar consulta y buscar por tokens relevantes
        tokens = _normalizar_tokens(query)

        if tokens:
            mask = pd.Series([False] * len(data), index=data.index)
            nombre = data['nombre'].astype(str).str.lower()
            descripcion = data['descripcion'].astype(str).str.lower()
            categoria_col = data['categoria'].astype(str).str.lower()
            for tok in tokens:
                mask = mask | nombre.str.contains(tok, na=False) | descripcion.str.contains(tok, na=False) | categoria_col.str.contains(tok, na=False)
            data = data[mask]

        # Si no hay resultados, devolver top por categoría o general
        if data.empty:
            if categoria:
                data = df_productos.copy()
                data = data[data['categoria'].astype(str).str.lower().str.contains(str(categoria).lower(), na=False)]
            else:
                data = df_productos.copy()

        # Convertir a formato de respuesta
        resultado = []
        for _, row in data.head(10).iterrows():
            resultado.append({
                "producto": row['nombre'],
                "categoria": row['categoria'],
                "precio": float(row['precio']),
                "vendedor": row['vendedor'],
                "stock": int(row['stock'])
            })
        return resultado
    
    # Fallback a datos simulados
    cypher = """
    MATCH (p:Producto)
    OPTIONAL MATCH (p)-[:VENDIDO_POR]->(v:Vendedor)
    WHERE $categoria IS NULL OR toLower(p.categoria) CONTAINS toLower($categoria)
    RETURN p.nombre as producto,
           p.categoria as categoria,
        p.precio as precio,
           v.id as vendedor
    ORDER BY p.precio
    LIMIT 10
    """
    return get_mock_data("producto")


def buscar_vendedor(_: str):
    """Busca vendedores del dataset real"""
    if df_vendedores is not None:
        resultado = []
        for _, row in df_vendedores.head(10).iterrows():
            resultado.append({
                "vendedor": row['nombre'],
                "ciudad": row['ciudad'],
                "calificacion": float(row['calificacion']),
                "productos_vendidos": int(row['productos_vendidos']),
                "especialidad": row['especialidad']
            })
        return resultado
    
    # Fallback
    return get_mock_data("vendedor")


def verificar_stock(_: str):
    """Verifica disponibilidad de productos del dataset real"""
    if df_productos is not None:
        resultado = []
        data = df_productos.sort_values('stock', ascending=False)
        for _, row in data.head(10).iterrows():
            resultado.append({
                "producto": row['nombre'],
                "disponible": int(row['stock']),
                "precio": float(row['precio']),
                "categoria": row['categoria']
            })
        return resultado
    
    # Fallback
    return get_mock_data("stock")


def consultar_clientes(_: str):
    """Consulta cantidad de clientes del dataset"""
    if df_productos is not None:
        total = len(df_productos)
        return [{"total_clientes": total}]
    
    # Fallback
    return get_mock_data("clientes")


def generar_respuesta(query: str, funcion: str, datos: list):
    """Genera respuesta usando Google Gemini con fallback robusto"""
    if not datos:
        return "No se encontraron resultados en la base de datos."

    # Preparar datos para el prompt
    datos_json = json.dumps(datos[:3], indent=2, ensure_ascii=False)
    
    # Si tenemos API key, intentar usar Gemini
    if GOOGLE_API_KEY:
        prompt = f"""Eres un asistente de e-commerce profesional y amigable en español.
El cliente preguntó: "{query}"
Se ejecutó la función: {funcion}

Datos encontrados:
{datos_json}

Responde de forma natural y concisa (máximo 3 líneas) mencionando los productos, precios o detalles encontrados."""

        try:
            # Seleccionar modelo disponible con generateContent
            modelo = None
            try:
                modelos = list(genai.list_models())
                candidatos = [m for m in modelos if "generateContent" in getattr(m, "supported_generation_methods", [])]
                preferidos = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]
                for nombre in preferidos:
                    for m in candidatos:
                        if m.name == nombre:
                            modelo = m.name
                            break
                    if modelo:
                        break
                if not modelo and candidatos:
                    modelo = candidatos[0].name
            except Exception:
                modelo = None

            if not modelo:
                return _generar_respuesta_fallback(query, funcion, datos)

            model = genai.GenerativeModel(modelo)
            response = model.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                return f"Se encontraron {len(datos)} resultados: {list(datos[0].values())[0]}"
        except Exception:
            # Fallback: generar respuesta simple
            return _generar_respuesta_fallback(query, funcion, datos)
    else:
        # Sin API key, usar fallback directamente
        return _generar_respuesta_fallback(query, funcion, datos)


def _generar_respuesta_fallback(query: str, funcion: str, datos: list):
    """Genera respuesta simple sin LLM"""
    try:
        if len(datos) == 0:
            return "No hay resultados disponibles."
        
        if funcion == "buscar_producto":
            categoria = extraer_categoria(query)
            ejemplos = []
            for d in datos[:3]:
                nombre = d.get("producto") or d.get("nombre") or next(iter(d.values()))
                # Si el nombre es numérico, presentarlo de forma más natural
                if isinstance(nombre, (int, float)) or (isinstance(nombre, str) and nombre.isdigit()):
                    nombre = f"Producto {nombre}"
                precio = d.get("precio")
                vendedor = d.get("vendedor")
                pieza = f"{nombre}"
                if precio is not None:
                    try:
                        precio_fmt = f"{float(precio):.2f}"
                    except Exception:
                        precio_fmt = str(precio)
                    pieza += f" (${precio_fmt})"
                if vendedor:
                    pieza += f" - {vendedor}"
                ejemplos.append(pieza)
            if categoria:
                return f"En la categoría {categoria} encontré {len(datos)} productos. Ejemplos: {'; '.join(ejemplos)}."
            return f"Sí, encontré {len(datos)} productos. Ejemplos: {'; '.join(ejemplos)}."

        if funcion == "buscar_vendedor":
            ejemplos = []
            for d in datos[:3]:
                nombre = d.get("vendedor") or d.get("nombre") or next(iter(d.values()))
                ciudad = d.get("ciudad")
                pieza = f"{nombre}"
                if ciudad:
                    pieza += f" ({ciudad})"
                ejemplos.append(pieza)
            return f"Encontré {len(datos)} vendedores. Ejemplos: {'; '.join(ejemplos)}."

        if funcion == "verificar_stock":
            ejemplos = []
            for d in datos[:3]:
                nombre = d.get("producto") or d.get("nombre") or next(iter(d.values()))
                disponible = d.get("disponible") or d.get("stock")
                pieza = f"{nombre}"
                if disponible is not None:
                    pieza += f" ({disponible} disponibles)"
                ejemplos.append(pieza)
            return f"Hay stock para {len(datos)} productos. Ejemplos: {'; '.join(ejemplos)}."

        if funcion == "consultar_clientes":
            total = None
            for d in datos:
                if "total_clientes" in d:
                    total = d.get("total_clientes")
                    break
            if total is not None:
                return f"Actualmente hay aproximadamente {total} clientes registrados."
            return f"Se encontraron {len(datos)} registros relacionados con clientes."

        # Fallback genérico
        first_item = datos[0]
        valores = list(first_item.values())
        return f"Se encontraron {len(datos)} resultados. Ejemplo: {valores[0]}"
    except Exception:
        return f"Consulta realizada. Se encontraron {len(datos)} resultados."


@app.post("/pedidos", response_model=PedidoResponse)
def crear_pedido(req: PedidoRequest):
    """Crea un nuevo pedido con los items proporcionados"""
    if not req.items or len(req.items) == 0:
        raise HTTPException(status_code=400, detail="Debe proporcionar al menos un item")
    
    # Generar ID único para el pedido
    pedido_id = f"PED-{len(pedidos_almacenados) + 1:05d}"
    timestamp = datetime.now().isoformat()
    
    # Calcular total
    total = sum(item.cantidad * item.precio for item in req.items)
    
    # Crear pedido
    pedido = {
        "id": pedido_id,
        "timestamp": timestamp,
        "items": [item.dict() for item in req.items],
        "total": round(total, 2),
        "estado": "completado"
    }
    
    # Almacenar pedido
    pedidos_almacenados.append(pedido)
    guardar_pedidos()
    
    return PedidoResponse(**pedido)


@app.get("/pedidos")
def obtener_pedidos():
    """Obtiene todos los pedidos realizados"""
    return {
        "total": len(pedidos_almacenados),
        "pedidos": pedidos_almacenados
    }


@app.get("/pedidos/{pedido_id}")
def obtener_pedido(pedido_id: str):
    """Obtiene un pedido específico por ID"""
    for pedido in pedidos_almacenados:
        if pedido["id"] == pedido_id:
            return pedido
    raise HTTPException(status_code=404, detail="Pedido no encontrado")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query vacia")

    # Intento de agregar al carrito por texto
    order_items = parse_order_intent(req.query)
    if order_items:
        producto = order_items[0].get("producto")
        cantidad = order_items[0].get("cantidad")
        answer = f"Agregué {cantidad} unidad(es) de {producto} al carrito."
        return {
            "query": req.query,
            "function": "agregar_carrito",
            "similitud": 1.0,
            "results": order_items,
            "answer": answer,
            "action": "add_to_cart",
            "cart_items": order_items,
        }

    funcion, similitud = seleccionar_funcion(req.query)

    if funcion == "buscar_producto":
        results = buscar_producto(req.query)
    elif funcion == "buscar_vendedor":
        results = buscar_vendedor(req.query)
    elif funcion == "consultar_clientes":
        results = consultar_clientes(req.query)
    else:
        results = verificar_stock(req.query)

    answer = generar_respuesta(req.query, funcion, results)

    return {
        "query": req.query,
        "function": funcion,
        "similitud": similitud,
        "results": results,
        "answer": answer,
    }

@app.get("/")
async def root():
    """Servir frontend desde raíz"""
    from fastapi.responses import FileResponse
    index_path = BASE_DIR / "front" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend no encontrado"}