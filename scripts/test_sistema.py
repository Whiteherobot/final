"""
Script de prueba rápida del sistema transaccional
Ejecuta: python test_sistema.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd

print("="*70)
print("TEST DEL SISTEMA TRANSACCIONAL")
print("="*70)

# 1. Cargar configuración
print("\n[1/5] Cargando configuración...")
BASE_DIR = Path.cwd()
if not (BASE_DIR / "data").exists() and (BASE_DIR.parent / "data").exists():
    BASE_DIR = BASE_DIR.parent

load_dotenv(BASE_DIR / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

print(f"  ✓ Neo4j URI: {NEO4J_URI}")
print(f"  ✓ Neo4j Usuario: {NEO4J_USER}")
print(f"  ✓ Google API: {'Configurada ✓' if GOOGLE_API_KEY else 'No configurada (opcional)'}")

# 2. Conectar a Neo4j
print("\n[2/5] Conectando a Neo4j...")
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        result.single()
    print("  ✓ Conexión exitosa a Neo4j!")
except Exception as e:
    print(f"  ✗ Error conectando a Neo4j: {e}")
    print("\n  Soluciones:")
    print("  1. Verifica que Neo4j esté corriendo (Neo4j Desktop o Docker)")
    print("  2. Verifica usuario y contraseña en el archivo .env")
    print("  3. Verifica que el puerto 7687 esté disponible")
    exit(1)

# 3. Cargar datasets
print("\n[3/5] Cargando datasets...")
try:
    df_productos = pd.read_csv(BASE_DIR / "data" / "dataset_transaccional.csv")
    df_vendedores = pd.read_csv(BASE_DIR / "data" / "dataset_vendedores.csv")
    print(f"  ✓ Productos: {len(df_productos)} registros")
    print(f"  ✓ Vendedores: {len(df_vendedores)} registros")
except FileNotFoundError as e:
    print(f"  ✗ Error: {e}")
    print("  Asegúrate de estar en la carpeta 'final' con los archivos CSV")
    exit(1)

# 4. Cargar datos a Neo4j
print("\n[4/5] Cargando datos a Neo4j...")
with driver.session() as session:
    # Limpiar base de datos
    session.run("MATCH (n) DETACH DELETE n")
    print("  ✓ Base de datos limpiada")
    
    # Cargar vendedores
    for _, v in df_vendedores.iterrows():
        query = """
        CREATE (v:Vendedor {
            id: $id, nombre: $nombre, email: $email, ciudad: $ciudad,
            calificacion: $calificacion, productos_vendidos: $productos_vendidos,
            especialidad: $especialidad
        })
        """
        session.run(query, v.to_dict())
    print(f"  ✓ {len(df_vendedores)} vendedores cargados")
    
    # Cargar productos
    for _, p in df_productos.iterrows():
        # Crear producto
        query_prod = """
        CREATE (p:Producto {
            id: $id, nombre: $nombre, categoria: $categoria,
            precio: $precio, stock: $stock, ubicacion: $ubicacion,
            descripcion: $descripcion
        })
        """
        session.run(query_prod, {
            'id': int(p['id']), 'nombre': p['nombre'],
            'categoria': p['categoria'], 'precio': float(p['precio']),
            'stock': int(p['stock']), 'ubicacion': p['ubicacion'],
            'descripcion': p['descripcion']
        })
        
        # Crear relación
        query_rel = """
        MATCH (p:Producto {id: $producto_id})
        MATCH (v:Vendedor {id: $vendedor_id})
        CREATE (p)-[:VENDIDO_POR]->(v)
        """
        session.run(query_rel, {
            'producto_id': int(p['id']),
            'vendedor_id': p['vendedor']
        })
    
    print(f"  ✓ {len(df_productos)} productos cargados con relaciones")

# 5. Verificar datos
print("\n[5/5] Verificando datos en Neo4j...")
with driver.session() as session:
    result = session.run("MATCH (n) RETURN count(n) as total")
    total = result.single()['total']
    print(f"  ✓ Total nodos en Neo4j: {total}")
    
    # Mostrar productos de ejemplo
    result = session.run("""
        MATCH (p:Producto)-[:VENDIDO_POR]->(v:Vendedor)
        RETURN p.nombre as producto, p.precio as precio, v.nombre as vendedor
        LIMIT 3
    """)
    
    print("\n  Muestra de datos:")
    for record in result:
        print(f"    • {record['producto']}: ${record['precio']} - {record['vendedor']}")

driver.close()

print("\n" + "="*70)
print("✅ SISTEMA INICIALIZADO CORRECTAMENTE")
print("="*70)
print("\nPróximos pasos:")
print("1. Abre el notebook: pipeline_transaccional_neo4j.ipynb")
print("2. Ejecuta las celdas en orden")
print("3. Prueba consultas como: '¿Cuánto cuesta el iPhone 14?'")
print("\nO visualiza en Neo4j Browser:")
print("  http://localhost:7474")
print("  Query: MATCH (n) RETURN n LIMIT 50")
print("="*70)
