# Guía de uso: Servicios de Inferencia RL para Unitree G1 y Go2

## 1. Resumen

Este documento describe cómo consumir los dos servicios de inferencia de locomoción desplegados en AWS:

| Servicio | Robot | Tipo | Observación (dim) | Acción (dim) |
|---|---|---|---|---|
| **G1** | Unitree G1 (bípedo) | Checkpoint TorchScript, LSTM recurrente | 47 floats | 12 floats |
| **Go2** | Unitree Go2 (cuadrúpedo) | Checkpoint TorchScript, ventana de historial | 45 floats | 12 floats |

Ambos servicios corren en contenedores independientes sobre AWS ECS Fargate (solo CPU, sin GPU), cada uno detrás de su propio Application Load Balancer. Son inferencia pura: reciben una observación y devuelven la acción predicha por la política pre-entrenada. No controlan hardware real ni ejecutan motores de bajo nivel — es responsabilidad del cliente aplicar la acción a su simulación o robot.

**URLs actuales (pueden cambiar si el servicio se redepliega):**

- G1: `http://unitree-rl-g1-alb-420244318.eu-central-1.elb.amazonaws.com`
- Go2: `http://unitree-rl-go2-alb-1880494698.eu-central-1.elb.amazonaws.com`

## 2. Endpoints (idénticos en ambos servicios)

### `GET /health`
Devuelve `200 OK` solo cuando el modelo ya está cargado en memoria.
```json
{"status": "ok"}
```

### `GET /metadata`
Describe el contrato exacto del servicio: nombre del robot, checkpoint usado (con su hash SHA-256), dimensiones de observación/acción, orden de las articulaciones, cómo escalar la observación de entrada y cómo interpretar la acción de salida.

### `POST /predict`
Punto central: envía una observación y recibe la acción predicha.

**Request:**
```json
{
  "observation": [0.0, 0.0, "... (47 floats para G1, 45 para Go2)"],
  "session_id": "opcional — ver sección 4"
}
```

**Response:**
```json
{
  "actions": [0.12, -0.05, "... (12 floats)"],
  "inference_latency_ms": 0.68,
  "checkpoint_id": "g1-motion-unitree_rl_gym-cf668f75b90d",
  "session_id": "2f53fe9cbcfa4a01a7d4357b5b58a616",
  "session_was_reset": true
}
```

Si `observation` no tiene exactamente la longitud correcta, o contiene un valor no finito (NaN/Inf), el servicio responde `422` con el detalle del error — nunca acepta observaciones con forma incorrecta.

### `DELETE /session/{session_id}`
Elimina manualmente el estado recurrente de una sesión antes de que expire por sí sola (10 minutos de inactividad).

## 3. Cómo construir la observación

Cada servicio espera la observación **ya pre-escalada** siguiendo su propia fórmula — no se envían valores crudos del robot. El layout exacto está en `/metadata`, y en resumen es:

**G1 (47 valores):**
| Rango | Contenido | Escala |
|---|---|---|
| 0:3 | Velocidad angular | × 0.25 |
| 3:6 | Vector de gravedad (orientación) | — |
| 6:9 | Comando de velocidad [vx, vy, yaw_rate] | × [2.0, 2.0, 0.25] |
| 9:21 | Posición articular − posición por defecto | × 1.0 |
| 21:33 | Velocidad articular | × 0.05 |
| 33:45 | Acción anterior (cruda) | — |
| 45:47 | [sin(fase), cos(fase)], periodo de marcha ≈0.8s | — |

**Go2 (45 valores):** igual estructura, sin los 2 últimos valores de fase (termina en 33:45 = acción anterior).

La acción que devuelve `/predict` es **cruda** (salida directa de la red). Para convertirla en el ángulo objetivo de cada articulación:

```
target_dof_pos = default_angles + action * action_scale   (action_scale = 0.25 en ambos)
```

Los valores exactos de `default_angles` y el orden de las articulaciones están en `/metadata` → `action_scaling` y `joint_order`.

## 4. Estado recurrente (`session_id`)

Ambos modelos mantienen memoria interna entre llamadas (G1: memoria LSTM; Go2: ventana de las últimas 5 observaciones). Por eso el contrato usa sesiones:

- **Primera llamada:** omitir `session_id` → el servicio crea una sesión nueva con memoria en cero y devuelve un `session_id` nuevo (`session_was_reset: true`).
- **Llamadas siguientes del mismo robot/episodio:** reenviar el `session_id` recibido → la memoria continúa correctamente (`session_was_reset: false`).
- **Nunca reutilizar el mismo `session_id` para dos robots distintos** — corromperían su memoria mutuamente.
- Si una sesión queda inactiva más de 10 minutos, se purga automáticamente; la siguiente llamada con ese id simplemente empieza una sesión nueva.

## 5. Ejemplos con `curl`

**G1 — nueva sesión:**
```bash
curl -X POST http://unitree-rl-g1-alb-420244318.eu-central-1.elb.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"observation": [0.0, 0.0, 0.0, "... completar hasta 47 valores"]}'
```

**G1 — continuar la misma sesión:**
```bash
curl -X POST http://unitree-rl-g1-alb-420244318.eu-central-1.elb.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"observation": [...], "session_id": "2f53fe9cbcfa4a01a7d4357b5b58a616"}'
```

**Go2 — metadata:**
```bash
curl http://unitree-rl-go2-alb-1880494698.eu-central-1.elb.amazonaws.com/metadata
```

## 6. Rendimiento esperado

Medido con carga sostenida real (50 req/s por servicio durante 10 minutos):

| Métrica | G1 | Go2 |
|---|---|---|
| Latencia de inferencia P99 (solo modelo) | 6.55 ms | 3.29 ms |
| Throughput sostenido | ~50 req/s | ~50 req/s |
| Errores | 0 | 0 |

La latencia **de red** (ida y vuelta HTTP completa) depende de dónde esté el cliente — es mucho menor si el cliente corre dentro de la misma región de AWS (`eu-central-1`) que si llama desde fuera.

## 7. Buenas prácticas para integradores

- Reutilizar la conexión HTTP (keep-alive) en vez de abrir una nueva por cada request — reduce la latencia de forma notable.
- Validar siempre la respuesta de `/health` antes de enviar tráfico real.
- No asumir el orden de las articulaciones de memoria: leerlo de `/metadata` en cada integración nueva, por si el checkpoint desplegado cambia.
- Un mismo `session_id` nunca debe compartirse entre dos instancias de robot corriendo en paralelo.
