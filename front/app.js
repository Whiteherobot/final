const apiUrl = "http://localhost:8000";

// DOM Elements
const queryInput = document.getElementById("query");
const sendButton = document.getElementById("send");
const statusEl = document.getElementById("status");
const answerEl = document.getElementById("answer");
const detailsEl = document.getElementById("details");
const functionLabel = document.getElementById("function-label");
const confidenceLabel = document.getElementById("confidence-label");
const resultadosSection = document.getElementById("resultados-section");
const productosLista = document.getElementById("productos-lista");

// Carrito
const carritoItems = document.getElementById("carrito-items");
const carritoResumen = document.getElementById("carrito-resumen");
const totalPrecio = document.getElementById("total-precio");
const btnComprar = document.getElementById("btn-comprar");
const historialPedidos = document.getElementById("historial-pedidos");

// Modal
const modal = document.getElementById("modal-confirmacion");
const modalResumen = document.getElementById("modal-resumen");
const btnCancelar = document.getElementById("btn-cancelar");
const btnConfirmar = document.getElementById("btn-confirmar");
const modalCerrar = document.querySelector(".modal-cerrar");

// Carrito de compras (estado global)
let carrito = [];

// Cargar historial al iniciar
cargarHistorial();

// EVENT LISTENERS
sendButton.addEventListener("click", sendQuery);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    sendQuery();
  }
});

btnComprar.addEventListener("click", mostrarConfirmacion);
btnCancelar.addEventListener("click", cerrarModal);
btnConfirmar.addEventListener("click", confirmarPedido);
modalCerrar.addEventListener("click", cerrarModal);
window.addEventListener("click", (e) => {
  if (e.target === modal) {
    cerrarModal();
  }
});

// BÚSQUEDA DE PRODUCTOS
async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query) {
    statusEl.textContent = "Escribe una consulta.";
    return;
  }

  sendButton.disabled = true;
  statusEl.textContent = "Consultando...";
  answerEl.textContent = "";
  productosLista.innerHTML = "";

  try {
    const res = await fetch(apiUrl + "/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });

    if (!res.ok) {
      const msg = await res.text();
      throw new Error(msg || "Error en la API");
    }

    const data = await res.json();
    answerEl.textContent = data.answer || "Sin respuesta.";
    
    functionLabel.textContent = `Función: ${data.function || "-"}`;
    const conf = typeof data.similitud === "number" ? data.similitud.toFixed(2) : "N/A";
    confidenceLabel.textContent = `Confianza: ${conf}`;
    
    // Mostrar detalles técnicos
    const details = {
      "Función seleccionada": data.function,
      "Similitud": data.similitud || "N/A",
      "Resultados encontrados": data.results.length,
      "Acción": data.action || "-",
      "Detalles": data.results
    };
    detailsEl.textContent = JSON.stringify(details, null, 2);

    // Si la respuesta indica agregar al carrito
    if (data.action === "add_to_cart" && Array.isArray(data.cart_items)) {
      addItemsToCart(data.cart_items);
    } else if (data.function === "agregar_carrito" && Array.isArray(data.results)) {
      addItemsToCart(data.results);
    }
    
    // Mostrar productos si es una búsqueda de producto
    if (data.function === "buscar_producto" && data.results && data.results.length > 0) {
      mostrarProductos(data.results);
      resultadosSection.style.display = "block";
    } else {
      resultadosSection.style.display = "none";
    }
    
    statusEl.textContent = "Listo.";
  } catch (err) {
    statusEl.textContent = "Error al consultar la API.";
    answerEl.textContent = "Error: " + String(err);
    detailsEl.textContent = String(err);
  } finally {
    sendButton.disabled = false;
  }
}

// MOSTRAR PRODUCTOS
function mostrarProductos(productos) {
  productosLista.innerHTML = "";
  
  productos.forEach((prod, idx) => {
    const nombre = prod.producto || prod.nombre || `Producto ${idx}`;
    const precio = parseFloat(prod.precio) || 0;
    const categoria = prod.categoria || "Sin categoría";
    
    const card = document.createElement("div");
    card.className = "producto-card";
    card.innerHTML = `
      <div class="producto-nombre">${nombre}</div>
      <div class="producto-precio">$${precio.toFixed(2)}</div>
      <div class="producto-controles">
        <input type="number" class="qty-${idx}" value="1" min="1" max="10">
        <button class="btn-agregar" onclick="agregarAlCarrito('${nombre}', ${precio}, ${idx})">Agregar</button>
      </div>
    `;
    productosLista.appendChild(card);
  });
}

// AGREGAR AL CARRITO
function agregarAlCarrito(nombre, precio, idx) {
  const cantidadInput = document.querySelector(`.qty-${idx}`);
  const cantidad = parseInt(cantidadInput.value) || 1;
  
  if (cantidad <= 0) {
    alert("Por favor ingresa una cantidad válida");
    return;
  }
  
  // Buscar si el producto ya existe en el carrito
  const itemExistente = carrito.find(item => item.producto === nombre && item.precio === precio);
  
  if (itemExistente) {
    itemExistente.cantidad += cantidad;
  } else {
    carrito.push({
      producto: nombre,
      precio: precio,
      cantidad: cantidad
    });
  }
  
  mostrarCarrito();
  mostrarNotificacion(`[OK] ${nombre} agregado al carrito`);
}

function addItemsToCart(items) {
  items.forEach((item) => {
    const nombre = item.producto;
    const precio = parseFloat(item.precio) || 0;
    const cantidad = parseInt(item.cantidad) || 1;

    const itemExistente = carrito.find((c) => c.producto === nombre && c.precio === precio);
    if (itemExistente) {
      itemExistente.cantidad += cantidad;
    } else {
      carrito.push({ producto: nombre, precio, cantidad });
    }
  });

  mostrarCarrito();
  if (items.length === 1) {
    mostrarNotificacion(`[OK] ${items[0].producto} agregado al carrito`);
  } else {
    mostrarNotificacion("[OK] Productos agregados al carrito");
  }
}

// MOSTRAR CARRITO
function mostrarCarrito() {
  const carritoVacio = carrito.length === 0;
  
  if (carritoVacio) {
    carritoItems.innerHTML = '<p class="carrito-vacio">El carrito está vacío</p>';
    carritoResumen.style.display = "none";
  } else {
    carritoItems.innerHTML = carrito.map((item, idx) => `
      <div class="carrito-item">
        <div class="carrito-item-info">
          <div class="carrito-item-nombre">${item.producto}</div>
          <div class="carrito-item-detalles">
            ${item.cantidad} × $${item.precio.toFixed(2)}
          </div>
        </div>
        <div class="carrito-item-subtotal">
          $${(item.cantidad * item.precio).toFixed(2)}
        </div>
        <button class="btn-eliminar" onclick="eliminarDelCarrito(${idx})">Eliminar</button>
      </div>
    `).join("");
    
    const total = carrito.reduce((sum, item) => sum + (item.cantidad * item.precio), 0);
    totalPrecio.textContent = `$${total.toFixed(2)}`;
    carritoResumen.style.display = "block";
  }
}

// ELIMINAR DEL CARRITO
function eliminarDelCarrito(idx) {
  const item = carrito[idx];
  carrito.splice(idx, 1);
  mostrarCarrito();
  mostrarNotificacion(`[REMOVED] ${item.producto} eliminado del carrito`);
}

// MOSTRAR MODAL DE CONFIRMACIÓN
function mostrarConfirmacion() {
  if (carrito.length === 0) {
    alert("El carrito está vacío");
    return;
  }
  
  // Generar resumen en el modal
  const total = carrito.reduce((sum, item) => sum + (item.cantidad * item.precio), 0);
  
  modalResumen.innerHTML = carrito.map(item => `
    <div class="modal-resumen-item">
      <span>${item.producto} (${item.cantidad}x)</span>
      <span>$${(item.cantidad * item.precio).toFixed(2)}</span>
    </div>
  `).join("") + `
    <div class="modal-resumen-total">
      <span>Total:</span>
      <span>$${total.toFixed(2)}</span>
    </div>
  `;
  
  modal.style.display = "block";
}

// CERRAR MODAL
function cerrarModal() {
  modal.style.display = "none";
}

// CONFIRMAR PEDIDO
async function confirmarPedido() {
  if (carrito.length === 0) {
    alert("El carrito está vacío");
    return;
  }
  
  btnConfirmar.disabled = true;
  btnConfirmar.textContent = "Procesando...";
  
  try {
    const res = await fetch(apiUrl + "/pedidos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        items: carrito
      })
    });
    
    if (!res.ok) {
      throw new Error("Error al procesar el pedido");
    }
    
    const pedido = await res.json();
    
    // Mostrar notificación de éxito
    mostrarNotificacion(`[SUCCESS] Pedido agregado con exito! ID: ${pedido.id}`);
    
    // Limpiar carrito
    carrito = [];
    mostrarCarrito();
    
    // Cerrar modal
    cerrarModal();
    
    // Recargar historial
    cargarHistorial();
    
  } catch (err) {
    alert("Error: " + String(err));
  } finally {
    btnConfirmar.disabled = false;
    btnConfirmar.textContent = "Confirmar Pedido";
  }
}

// CARGAR Y MOSTRAR HISTORIAL DE PEDIDOS
async function cargarHistorial() {
  try {
    const res = await fetch(apiUrl + "/pedidos");
    if (!res.ok) throw new Error("Error al cargar historial");
    
    const data = await res.json();
    const pedidos = data.pedidos || [];
    
    if (pedidos.length === 0) {
      historialPedidos.innerHTML = '<p class="sin-pedidos">No hay pedidos realizados aún.</p>';
    } else {
      historialPedidos.innerHTML = pedidos.map(pedido => {
        const fecha = new Date(pedido.timestamp).toLocaleString('es-ES');
        const itemsHTML = pedido.items.map(item => 
          `${item.producto} (${item.cantidad}x $${item.precio.toFixed(2)})`
        ).join(", ");
        
        return `
          <div class="pedido-card">
            <div class="pedido-header">
              <span class="pedido-id">${pedido.id}</span>
              <span class="pedido-fecha">${fecha}</span>
            </div>
            <div class="pedido-items-list">
              ${itemsHTML}
            </div>
            <div class="pedido-total">
              <span>Total:</span>
              <span>$${pedido.total.toFixed(2)}</span>
            </div>
          </div>
        `;
      }).join("");
    }
  } catch (err) {
    historialPedidos.innerHTML = '<p class="sin-pedidos">Error al cargar historial</p>';
    console.error("Error cargando historial:", err);
  }
}

// NOTIFICACIÓN
function mostrarNotificacion(mensaje) {
  const notif = document.createElement("div");
  notif.className = "notificacion";
  notif.textContent = mensaje;
  document.body.appendChild(notif);
  
  setTimeout(() => {
    notif.classList.add("saliendo");
    setTimeout(() => notif.remove(), 300);
  }, 3000);
}
