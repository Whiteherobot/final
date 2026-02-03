"""
Tabla de Funciones para Sistema Transaccional
Mapea queries de usuario a funciones específicas usando embeddings
"""

FUNCIONES_SISTEMA = [
    {
        "id": 1,
        "nombre_funcion": "buscar_producto",
        "descripcion": "Busca productos en el inventario por nombre, categoria o caracteristicas. Retorna: stock, precio, vendedor, ubicacion",
        "query_examples": [
            "¿Cuanto cuesta el iPhone 14?",
            "¿Que laptops tienen en stock?",
            "Busco audifonos con cancelacion de ruido",
            "¿Tienen Mouse Logitech disponible?",
            "Productos de electronica en Lima"
        ]
    },
    {
        "id": 2,
        "nombre_funcion": "buscar_vendedor",
        "descripcion": "Encuentra vendedores por ciudad, especialidad o calificacion. Retorna: nombre, email, ciudad, calificacion, especialidad",
        "query_examples": [
            "¿Quien vende smartphones en Lima?",
            "Vendedores con calificacion alta",
            "¿Que vendedor tiene MacBook?",
            "Vendedores especializados en electronica",
            "¿Hay vendedores en Arequipa?"
        ]
    },
    {
        "id": 3,
        "nombre_funcion": "verificar_stock",
        "descripcion": "Verifica disponibilidad y cantidad de productos. Retorna: stock_actual, precio, ubicacion_almacen",
        "query_examples": [
            "¿Cuantos iPhone 14 hay disponibles?",
            "Stock de Samsung Galaxy S23",
            "¿Hay suficiente stock de cables HDMI?",
            "Disponibilidad de Monitor LG",
            "¿Cuantas unidades quedan del MacBook?"
        ]
    },
    {
        "id": 4,
        "nombre_funcion": "comparar_precios",
        "descripcion": "Compara precios entre productos similares o del mismo vendedor. Retorna: producto, precio, vendedor, diferencia_precio",
        "query_examples": [
            "¿Que smartphone es mas barato?",
            "Comparar precios de laptops",
            "¿iPhone o Samsung cual cuesta menos?",
            "Diferencia de precio entre monitores",
            "¿Que vendedor tiene mejores precios?"
        ]
    },
    {
        "id": 5,
        "nombre_funcion": "crear_pedido",
        "descripcion": "Crea un nuevo pedido de compra. Retorna: pedido_id, total, fecha_entrega_estimada, vendedor",
        "query_examples": [
            "Quiero comprar un iPhone 14",
            "Crear pedido de Mouse Logitech",
            "Comprar 2 cables HDMI",
            "Agregar MacBook Pro al carrito",
            "Hacer pedido de audifonos Sony"
        ]
    },
    {
        "id": 6,
        "nombre_funcion": "consultar_categoria",
        "descripcion": "Lista productos por categoria. Retorna: productos, cantidad, rango_precios",
        "query_examples": [
            "¿Que productos de Audio tienen?",
            "Mostrar todos los accesorios",
            "Productos en categoria Electronica",
            "¿Que hay en Almacenamiento?",
            "Lista de productos de Oficina"
        ]
    },
    {
        "id": 7,
        "nombre_funcion": "buscar_por_ubicacion",
        "descripcion": "Filtra productos y vendedores por ciudad. Retorna: productos_disponibles, vendedores, tiempos_entrega",
        "query_examples": [
            "Productos disponibles en Lima",
            "¿Que venden en Arequipa?",
            "Vendedores en Cusco",
            "Stock en Lima",
            "¿Que hay disponible en mi ciudad?"
        ]
    },
    {
        "id": 8,
        "nombre_funcion": "obtener_recomendaciones",
        "descripcion": "Sugiere productos relacionados o complementarios. Retorna: productos_sugeridos, razon_recomendacion",
        "query_examples": [
            "¿Que me recomiendas con una laptop?",
            "Productos relacionados con iPhone",
            "¿Que accesorios van bien con este monitor?",
            "Sugerencias para mi setup de oficina",
            "¿Que mas necesito para gaming?"
        ]
    }
]

# Generar embeddings para cada funcion (se calcularan en runtime)
COLUMNAS_TABLA = ["id", "nombre_funcion", "descripcion", "embedding", "query_examples"]
