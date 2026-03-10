# Database

El servicio guarda metadata en PostgreSQL y evidencia en MinIO.

## Tablas

1. `sources`
   - catalogo logico de camaras, DVR o fuentes
2. `sessions`
   - una fila por conexion WebSocket autenticada
3. `state_snapshots`
   - resumen periodico de conteos, label dominante y latencia
4. `detection_events`
   - evento disparado por el label configurado con:
     - `clip_bucket`
     - `clip_object_key`
     - `preview_bucket`
     - `preview_object_key`
     - `annotation`

## Regla de almacenamiento

- PostgreSQL: metadata y trazabilidad
- MinIO: `preview.jpg` y `clip.mp4`
- Las URLs firmadas se generan al consultar `GET /events`

## Stack local por defecto

El `docker-compose.yml` de este repo levanta:

- `postgres`
- `minio`
- `websocket-api`

Ademas conecta `websocket-api` a la red externa
`invoice-route-app_app_network` para integrarlo con el stack principal si esa
red ya existe.

## Stack standalone

Si necesitas levantar todo aislado dentro de este repo, usa
`docker-compose.standalone.yml`. Ese archivo levanta:

- `postgres`
- `minio`
- `websocket-api`

No depende de redes externas ni de otros contenedores.
