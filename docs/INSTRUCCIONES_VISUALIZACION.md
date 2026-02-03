"""
Celda adicional para VISUALIZACIÓN DE NODOS en Neo4j
Agregar esta celda después de la celda "5. Cargar Datos a Neo4j"
"""

# CELDA NUEVA: Visualización del Grafo Neo4j

## Opción 1: Visualización Interactiva con pyvis

```python
# Instalar pyvis si no está instalado
try:
    from pyvis.network import Network
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvis"])
    from pyvis.network import Network

import networkx as nx

def visualizar_grafo_neo4j(neo4j_conn, output_file="grafo_neo4j.html"):
    """
    Crea una visualización interactiva del grafo de Neo4j
    """
    # Obtener todos los nodos
    query_nodos = """
    MATCH (n)
    RETURN id(n) as id, labels(n) as labels, properties(n) as props
    """
    nodos = neo4j_conn.query(query_nodos)
    
    # Obtener todas las relaciones
    query_relaciones = """
    MATCH (n)-[r]->(m)
    RETURN id(n) as source, id(m) as target, type(r) as type
    """
    relaciones = neo4j_conn.query(query_relaciones)
    
    # Crear red con pyvis
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=100)
    
    # Agregar nodos
    for nodo in nodos:
        node_id = nodo['id']
        label_type = nodo['labels'][0] if nodo['labels'] else 'Unknown'
        props = nodo['props']
        
        # Título con información del nodo
        if label_type == 'Producto':
            titulo = f"{props.get('nombre', 'N/A')}\\nPrecio: ${props.get('precio', 'N/A')}\\nStock: {props.get('stock', 'N/A')}"
            color = '#FF6B6B'  # Rojo para productos
        elif label_type == 'Vendedor':
            titulo = f"{props.get('nombre', 'N/A')}\\nCiudad: {props.get('ciudad', 'N/A')}\\nCalificación: {props.get('calificacion', 'N/A')}"
            color = '#4ECDC4'  # Turquesa para vendedores
        else:
            titulo = str(props)
            color = '#95E1D3'
        
        net.add_node(
            node_id,
            label=props.get('nombre', label_type),
            title=titulo,
            color=color,
            size=25
        )
    
    # Agregar relaciones
    for rel in relaciones:
        net.add_edge(
            rel['source'],
            rel['target'],
            title=rel['type'],
            label=rel['type'],
            color='#888888'
        )
    
    # Configurar física
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 100
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)
    
    # Guardar y mostrar
    net.show(output_file)
    print(f"✓ Grafo guardado en: {output_file}")
    print(f"  - Nodos: {len(nodos)}")
    print(f"  - Relaciones: {len(relaciones)}")
    print(f"\\nAbre el archivo '{output_file}' en tu navegador para ver el grafo interactivo.")
    
    return net

# Ejecutar visualización
visualizar_grafo_neo4j(neo4j_conn)
```

## Opción 2: Visualización Simple con NetworkX y Matplotlib

```python
import matplotlib.pyplot as plt
import networkx as nx

def visualizar_grafo_simple(neo4j_conn):
    """
    Crea una visualización simple con matplotlib
    """
    # Obtener datos
    query = """
    MATCH (n)-[r]->(m)
    RETURN n.nombre as source, m.nombre as target, type(r) as relation
    """
    resultados = neo4j_conn.query(query)
    
    # Crear grafo NetworkX
    G = nx.DiGraph()
    
    for res in resultados:
        G.add_edge(res['source'], res['target'], relation=res['relation'])
    
    # Visualizar
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                           node_size=3000, alpha=0.9)
    
    # Dibujar aristas
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                           arrows=True, arrowsize=20, 
                           arrowstyle='->', width=2)
    
    # Dibujar etiquetas de nodos
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Dibujar etiquetas de relaciones
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title("Grafo de Conocimiento - Neo4j", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Grafo visualizado con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas")

# Ejecutar visualización
visualizar_grafo_simple(neo4j_conn)
```

## Opción 3: Consultar y Ver Nodos en Tabla

```python
def mostrar_nodos_tabla(neo4j_conn):
    """
    Muestra los nodos en formato de tabla con pandas
    """
    # Ver productos
    print("=== PRODUCTOS ===")
    query_productos = """
    MATCH (p:Producto)-[:VENDIDO_POR]->(v:Vendedor)
    RETURN p.nombre as Producto, p.precio as Precio, p.stock as Stock,
           p.categoria as Categoria, v.nombre as Vendedor, p.ubicacion as Ubicacion
    ORDER BY p.precio DESC
    """
    df_productos = pd.DataFrame(neo4j_conn.query(query_productos))
    display(df_productos)
    
    print("\\n=== VENDEDORES ===")
    query_vendedores = """
    MATCH (v:Vendedor)
    RETURN v.nombre as Vendedor, v.ciudad as Ciudad, 
           v.calificacion as Calificación, v.especialidad as Especialidad
    ORDER BY v.calificacion DESC
    """
    df_vendedores = pd.DataFrame(neo4j_conn.query(query_vendedores))
    display(df_vendedores)
    
    # Resumen
    print("\\n=== RESUMEN DEL GRAFO ===")
    query_resumen = """
    MATCH (n)
    RETURN labels(n)[0] as Tipo, count(n) as Cantidad
    """
    df_resumen = pd.DataFrame(neo4j_conn.query(query_resumen))
    display(df_resumen)

# Ejecutar visualización en tabla
mostrar_nodos_tabla(neo4j_conn)
```

## Opción 4: Neo4j Browser (Recomendado para exploración)

```python
print("=== USAR NEO4J BROWSER ===" )
print("\\n1. Abre tu navegador y ve a: http://localhost:7474")
print("2. Conecta con tus credenciales:")
print(f"   - Usuario: {NEO4J_USER}")
print("   - Password: [tu contraseña]")
print("\\n3. Ejecuta estas queries en el browser:")
print("\\n   Ver todo el grafo:")
print("   MATCH (n) RETURN n LIMIT 50")
print("\\n   Ver solo productos:")
print("   MATCH (p:Producto)-[r:VENDIDO_POR]->(v:Vendedor) RETURN p, r, v")
print("\\n   Ver vendedores y sus productos:")
print("   MATCH (v:Vendedor)<-[:VENDIDO_POR]-(p:Producto) RETURN v, p")
```
