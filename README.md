# YOLO WebSocket Service

Servicio aislado para desplegar inferencia YOLO por WebSocket sobre FastAPI.

Esta carpeta ya puede vivir como repo independiente.

Pensado para este flujo:

1. La camara o celular abre una conexion WebSocket segura (`wss://`).
2. El cliente se autentica una sola vez con usuario y contrasena.
3. El cliente envia frames en JPEG.
4. El servidor responde con detecciones JSON en tiempo real.

## Estructura

- `app/`: servicio FastAPI, autenticacion y runtime de inferencia.
- `hash_password.py`: genera un hash PBKDF2 para no guardar la contrasena en claro.
- `.env.example`: variables listas para copiar como `.env`.
- `.env.production.example`: ejemplo realista para PostgreSQL + MinIO.
- `Dockerfile`: imagen lista para DigitalOcean.
- `DATABASE.md`: diseno de la base operativa para PostgreSQL + MinIO.
- `DEPLOY_DIGITALOCEAN.md`: comandos exactos para `doctl`.
- `digitalocean/app-spec.yaml`: opcional, solo si quieres automatizar App Platform por CLI.
- `examples/`: opcional, solo para probar el servicio desde Python o navegador.

## Protocolo WebSocket

Ruta:

```text
/ws/infer
```

El backend puede publicar uno o varios modelos sobre el mismo socket.
La seleccion recomendada es por `model_id` en el mensaje `auth`.
No hace falta exponer un endpoint distinto por modelo salvo que quieras
aislar despliegue, costos o escalado.

Compatibilidad actual con `vision-app`:

- esa web envia `model`, `model_id`, `model_selection` y `requested_model`
- normalmente manda `"1"` para el modelo general y `"2"` para el modelo de llenado
- este backend ya acepta esos campos y asigna aliases numericos automaticamente segun el orden de los modelos

### 1. Mensaje de autenticacion

El primer mensaje debe ser JSON:

```json
{
  "type": "auth",
  "username": "camera-ingenio-01",
  "password": "tu-clave",
  "source_id": "camara1",
  "source_name": "Camara 1",
  "model_id": "cana"
}
```

Antes del `auth`, el servidor manda un `hello` con los modelos disponibles:

```json
{
  "type": "hello",
  "protocol": "yolo-ws-v1",
  "message": "Send auth message first.",
  "default_model_id": "default",
  "available_models": [
    {
      "id": "default",
      "name": "Modelo General",
      "selector": "1",
      "file_name": "model.pt",
      "device": "cpu"
    },
    {
      "id": "cana",
      "name": "Modelo Cana",
      "selector": "2",
      "file_name": "cana.pt",
      "device": "cpu"
    }
  ]
}
```

Respuesta exitosa:

```json
{
  "type": "auth_ok",
  "session_id": "4f779adf7f5448c5a68f10ae8c651e80",
  "connection_id": "4f779adf",
  "source_id": "dvr-canal-1",
  "source_type": "rtsp",
  "source_name": "DVR Canal 1",
  "model": {
    "id": "cana",
    "name": "Modelo Cana",
    "file_name": "cana.pt",
    "device": "cpu"
  },
  "default_model_id": "default"
}
```

Si quieres cambiar de modelo desde la web, la forma mas limpia es cerrar
esa conexion y abrir otra con otro `model_id`. Asi cada sesion queda
registrada con su modelo real.

Para distinguir varias conexiones al mismo tiempo:

- usa un `source_id` estable por camara, por ejemplo `camara1`, `camara2`, `patio-norte`
- usa `source_name` para un nombre legible
- el backend tambien devuelve `connection_id`, un identificador corto por sesion

### 2. Envio de frames

Opcion recomendada para Python/OpenCV: enviar el frame como binario JPEG.

El servidor responde:

```json
{
  "type": "inference",
  "frame_id": "5c7f5f2ccf944f6dbfa0b2c4ec3e7eb5",
  "latency_ms": 38.42,
  "image": {
    "width": 1280,
    "height": 720
  },
  "counts": {
    "total": 2,
    "by_label": {
      "cana_buena": 1,
      "cana_danada": 1
    }
  },
  "detections": [
    {
      "class_id": 0,
      "label": "cana_buena",
      "confidence": 0.92,
      "xyxy": [120.5, 80.2, 400.1, 510.6]
    }
  ]
}
```

Tambien puedes enviar JSON si el cliente no maneja binarios:

```json
{
  "type": "frame",
  "frame_id": "frame-001",
  "image_b64": "<jpeg-base64>",
  "return_image": false
}
```

Mensajes auxiliares:

- `{"type":"ping"}` -> `{"type":"pong"}`
- `GET /models` -> lista los modelos disponibles y el `default_model_id`

## Flujo de evento real

El servicio ya soporta este flujo:

1. El cliente se autentica con un `source_id`.
2. El cliente manda frames desde camara, RTSP o video.
3. Si aparece una deteccion con label `25`, se dispara un evento.
4. El servicio guarda:
   - una imagen anotada en MinIO
   - los siguientes 5 segundos en video en MinIO
   - un registro en PostgreSQL
   - snapshots periodicos de estado
   - URLs firmadas temporales al consultar eventos

El upload a MinIO ocurre cuando el evento se cierra y se persiste dentro de
`app/events.py`, en la rutina `_finalize_event`.

Para el modelo `cana`, si el llenado detectado supera
`YOLO_WS_TELEGRAM_FILL_THRESHOLD`, el backend reutiliza este mismo pipeline
de eventos para guardar:

- una imagen anotada
- 5 segundos de clip
- ambos bajo el prefijo `YOLO_WS_FILL_EVENT_STORAGE_PREFIX` en MinIO
- y al cerrar el evento envia por Telegram la foto anotada y el video directamente

Por defecto:

- trigger: `YOLO_WS_TRIGGER_LABELS=25`
- clip: `YOLO_WS_CLIP_SECONDS=5`
- snapshots: `YOLO_WS_SNAPSHOT_INTERVAL_SECONDS=10`
- DB: `YOLO_WS_DATABASE_URL`
- object storage: `YOLO_WS_STORAGE_BACKEND=minio`

## Docker Compose

- `docker-compose.yml`: base limpia para despliegue. Levanta `postgres`,
  `minio` y `websocket-api` sin fijar `container_name` ni `ports`.
- `docker-compose.override.yml`: override local automatico. Expone puertos,
  fija nombres de contenedor y conecta la red externa local si existe.

## Variables importantes

- `YOLO_WS_AUTH_USERNAME`: usuario permitido.
- `YOLO_WS_AUTH_PASSWORD_HASH`: hash PBKDF2 recomendado.
- `YOLO_WS_AUTH_PASSWORD`: solo para pruebas rapidas.
- `YOLO_WS_DEFAULT_MODEL_ID`: modelo por defecto del socket.
- `YOLO_WS_MODELS_FILE`: archivo JSON con la lista de modelos y sus opciones. Es la forma recomendada para crecer a N modelos.
- `selector` en `YOLO_WS_MODELS_FILE`: valor recomendado para que los clientes llamen ese modelo. Puede ser `1`, `2`, `10` o el que quieras.
- `YOLO_WS_MODEL_PATH`: ruta local al `.pt`, relativa a esta misma carpeta.
- `YOLO_WS_MODEL_NAME`: nombre visible del modelo unico por compatibilidad.
- `YOLO_WS_MODEL_URL`: URL para descargar el modelo si no existe en disco.
- `YOLO_WS_MODEL_SHA256`: checksum opcional para validar la descarga.
- `aliases` en `YOLO_WS_MODELS_FILE`: aliases opcionales por modelo. Los numericos (`1`, `2`, `3`, ...) se agregan automaticamente segun el orden.
- `fill_events_enabled` en `YOLO_WS_MODELS_FILE`: activa la logica de llenado para ese modelo.
- `fill_event_storage_prefix` en `YOLO_WS_MODELS_FILE`: prefijo de almacenamiento para eventos de llenado de ese modelo.
- `YOLO_WS_MODEL_IDS`: activa modo multi-modelo, por ejemplo `default,cana`.
- `YOLO_WS_MODEL_<ID>_PATH`: ruta local del modelo `ID`.
- `YOLO_WS_MODEL_<ID>_URL`: URL opcional para descargar el modelo `ID`.
- `YOLO_WS_MODEL_<ID>_SHA256`: checksum opcional del modelo `ID`.
- `YOLO_WS_MODEL_<ID>_NAME`: nombre visible del modelo `ID` para UI.
- `YOLO_WS_MODEL_SELECTION_ALIASES`: aliases del selector que llega desde clientes externos, por ejemplo `1:default,2:cana`.
- `PORT`: puerto de DigitalOcean/App Platform.
- `YOLO_WS_TRIGGER_LABELS`: labels que disparan evento.
- `YOLO_WS_CLIP_SECONDS`: segundos de video que se guardan.
- `YOLO_WS_DATABASE_URL`: conexion a PostgreSQL. Acepta `postgresql+psycopg://`, `postgresql://` o `postgres://`.
- `YOLO_WS_DB_PATH`: fallback local para desarrollo si no defines `YOLO_WS_DATABASE_URL`.
- `YOLO_WS_SNAPSHOT_INTERVAL_SECONDS`: cada cuantos segundos guardar resumen.
- `YOLO_WS_SNAPSHOT_SAVE_EMPTY`: si guarda snapshots aun sin detecciones.
- `YOLO_WS_TELEGRAM_ENABLED`: activa alertas por Telegram.
- `YOLO_WS_TELEGRAM_BOT_TOKEN`: token del bot.
- `YOLO_WS_TELEGRAM_CHAT_ID`: chat destino.
- `YOLO_WS_TELEGRAM_MODEL_IDS`: modelos que disparan esta logica, por ejemplo `cana`.
- `YOLO_WS_TELEGRAM_FILL_THRESHOLD`: umbral estricto de llenado. Si el porcentaje detectado es mayor a este valor, envia alerta.
- `YOLO_WS_FILL_EVENT_STORAGE_PREFIX`: prefijo MinIO para estos eventos de llenado, por ejemplo `cana`.
- `YOLO_WS_STORAGE_BACKEND`: `minio` o `local`.
- `YOLO_WS_MINIO_*`: credenciales y bucket del object storage.
- `YOLO_WS_PRESIGNED_URL_EXPIRY_SECONDS`: vigencia de las URLs firmadas.
- `YOLO_WS_TEMP_DIR`: carpeta temporal antes de subir el clip.

## Base de datos

Para produccion se eligio:

1. PostgreSQL para metadata operativa
2. MinIO para archivos grandes

La BD guarda metadata, no binarios.
Los eventos guardan `bucket` y `object_key`.
Las `presigned URL` se generan al consultar la API.

Tablas operativas:

1. `sources`
2. `sessions`
3. `state_snapshots`
4. `detection_events`

Diseno detallado en `DATABASE.md`.

## Como generar el hash de la contrasena

```bash
python hash_password.py --password "tu-clave-segura"
```

## Ejecucion local

```bash
copy .env.example .env
python -m uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

## Docker local

```bash
docker build -t yolo-websocket .
docker run --rm -p 8000:8000 --env-file .env yolo-websocket
```

Para pruebas locales sin PostgreSQL ni MinIO puedes usar:

```env
YOLO_WS_STORAGE_BACKEND=local
YOLO_WS_DB_PATH=runtime/service.db
```

Ejemplo multi-modelo:

```env
YOLO_WS_DEFAULT_MODEL_ID=default
YOLO_WS_MODELS_FILE=config/models.json
```

```json
{
  "default_model_id": "default",
  "models": [
    {
      "id": "default",
      "name": "Modelo General",
      "selector": "1",
      "path": "runtime/model.pt"
    },
    {
      "id": "cana",
      "name": "Modelo Cana",
      "selector": "2",
      "path": "runtime/cana.pt",
      "fill_events_enabled": true,
      "fill_event_storage_prefix": "cana"
    }
  ]
}
```

Ejemplo con alerta Telegram para `cana`:

```env
YOLO_WS_TELEGRAM_ENABLED=true
YOLO_WS_TELEGRAM_FILL_THRESHOLD=50
YOLO_WS_TELEGRAM_BOT_TOKEN=...
YOLO_WS_TELEGRAM_CHAT_ID=...
```

## Consumir el servicio

### Cliente Python con camara

Requiere:

```bash
pip install websockets opencv-python
```

Uso:

```bash
python examples/python_camera_client.py ^
  --url ws://localhost:8000/ws/infer ^
  --username camera-ingenio-01 ^
  --password tu-clave ^
  --model-id cana ^
  --source-id camera-local-01 ^
  --source-name "Camara Local" ^
  --camera 0 ^
  --interval-ms 300 ^
  --show
```

### Cliente navegador o celular

Abre `examples/browser_camera_client.html`.

Ese ejemplo ya carga la lista de modelos desde `hello` y manda el
`model_id` seleccionado en `auth`.
Tambien permite mandar `source_id` y `source_name` para identificar la camara.

Notas:

1. Si quieres usar la camara del celular, sirve ese HTML por `https://` o desde `http://localhost`, porque los navegadores modernos restringen `getUserMedia`.
2. Si el backend esta en produccion, usa `wss://tu-dominio/ws/infer`.

### DVR, RTSP o video

Tu DVR normalmente no se conecta al WebSocket de forma nativa.
Lo correcto es usar un puente:

```text
DVR/RTSP/video -> script Python/OpenCV -> WebSocket -> YOLO
```

Ejemplo:

```bash
pip install websockets opencv-python
python examples/rtsp_video_bridge.py ^
  --url ws://localhost:8000/ws/infer ^
  --username camera-ingenio-01 ^
  --password tu-clave ^
  --model-id cana ^
  --source "rtsp://usuario:clave@IP:554/Streaming/Channels/101" ^
  --source-id dvr-canal-1 ^
  --source-name "DVR Canal 1"
```

Tambien sirve con un archivo:

```bash
python examples/rtsp_video_bridge.py ^
  --url ws://localhost:8000/ws/infer ^
  --username camera-ingenio-01 ^
  --password tu-clave ^
  --model-id cana ^
  --source "C:\\videos\\prueba.mp4" ^
  --source-id video-prueba ^
  --source-name "Video de Prueba"
```

## Despliegue en DigitalOcean

Opciones practicas:

1. Droplet con Docker: construyes la imagen y la levantas con `docker run`.
2. App Platform: usas este `Dockerfile` y defines las variables de entorno en el panel.

Recomendacion operativa:

- Si vas con `PostgreSQL + MinIO`, puedes usar `Droplet` o `App Platform`.
- El filesystem local sigue siendo temporal; por eso el servicio solo lo usa para generar el clip antes de subirlo a MinIO.

Ya viene lista como repo aparte. Si quieres correr el stack completo con PostgreSQL
y MinIO incluidos:

```bash
copy .env.example .env
docker compose up -d --build
```

Solo necesitas ajustar:

1. `app/`
2. `Dockerfile`
3. `requirements.txt`
4. `.env.example`
5. Tu modelo en `runtime/model.pt` o una `YOLO_WS_MODEL_URL`

### App Platform con spec

Esto es opcional. Solo sirve si quieres crear la app con `doctl` en vez del panel web.

Edita `digitalocean/app-spec.yaml` y cambia:

1. `YOUR_GITHUB_USER/YOUR_REPOSITORY`
2. `REPLACE_WITH_HASH`
3. `YOLO_WS_MODEL_URL`
4. `YOLO_WS_DATABASE_URL`
5. `YOLO_WS_MINIO_ENDPOINT`
6. `YOLO_WS_MINIO_ACCESS_KEY`
7. `YOLO_WS_MINIO_SECRET_KEY`
8. `YOLO_WS_MINIO_BUCKET`

Luego despliegas con `doctl`:

```bash
doctl apps create --spec digitalocean/app-spec.yaml
```

DigitalOcean documenta que App Platform acepta `dockerfile_path` en el app spec y que el servicio debe escuchar en `0.0.0.0` sobre el `http_port`. Tambien puedes definir `health_check.http_path` como `/healthz`. Fuentes oficiales:

- https://docs.digitalocean.com/products/app-platform/reference/dockerfile/
- https://docs.digitalocean.com/products/app-platform/reference/app-spec/
- https://docs.digitalocean.com/products/app-platform/how-to/manage-services/index.html
- https://docs.digitalocean.com/products/app-platform/how-to/manage-health-checks/

Recomendacion para el modelo:

1. Sube el `.pt` a DigitalOcean Spaces o a un almacenamiento privado accesible por URL.
2. Define `YOLO_WS_MODEL_URL`.
3. Opcionalmente define `YOLO_WS_MODEL_SHA256`.

Asi la imagen queda pequena y no dependes de subir el modelo dentro del repo.

## Notas de operacion

- Usa `wss://` en produccion, terminando TLS en Nginx, Caddy o el balanceador de DigitalOcean.
- Manten una sola replica por contenedor si usas CPU, para no cargar varias veces el modelo.
- Si vas a cargar muchos modelos pesados, separar por servicio puede ser mejor que tenerlos todos en memoria.
- Si conectas varias camaras, escala horizontalmente en vez de subir workers de Uvicorn.
- Puedes consultar la base operativa con `GET /db/stats`, `GET /sources`, `GET /sessions`, `GET /snapshots` y `GET /events`.
- `GET /events` devuelve `clip_url` y `preview_url` como `presigned URL` cuando el backend es MinIO.
