# Deploy en DigitalOcean

## Opcion 1: Droplet con Docker

1. clona este repo en el Droplet
2. copia `.env.example` a `.env`
3. genera el hash:

```bash
python hash_password.py --password "tu-clave"
```

4. coloca tu modelo en `runtime/model.pt` o define `YOLO_WS_MODEL_URL`
5. levanta el stack:

```bash
docker compose -f docker-compose.yml up -d --build
```

Servicios segun los puertos que definas en `.env`:

- API: `http://IP:${YOLO_WS_PORT}`
- MinIO API: `http://IP:${MINIO_API_PORT}`
- MinIO Console: `http://IP:${MINIO_CONSOLE_PORT}`
- PostgreSQL: `IP:${POSTGRES_PORT}`

## Opcion 2: separar infraestructura gestionada

Si luego quieres pasar a servicios gestionados:

- dejas `websocket-api` en contenedor
- cambias `YOLO_WS_DATABASE_URL` a PostgreSQL gestionado
- cambias `YOLO_WS_MINIO_*` a MinIO/Spaces externo

## Recomendacion

Para un despliegue simple e independiente, usa Droplet con `docker compose`.
