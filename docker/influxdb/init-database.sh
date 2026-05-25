#!/bin/sh
URL="${INFLUX_URL:-http://influxdb:8181}"
DB="${INFLUX_DATABASE:-biofizic}"

echo "Astept InfluxDB la ${URL}..."
i=0
while [ "$i" -lt 30 ]; do
  if curl -sf "${URL}/health" >/dev/null 2>&1; then
    break
  fi
  i=$((i + 1))
  sleep 2
done

echo "Creez database '${DB}'..."
curl -sf -X POST "${URL}/api/v3/configure/database" \
  -H "Content-Type: application/json" \
  -d "{\"db\":\"${DB}\"}" \
  || curl -sf -X POST "${URL}/api/v3/configure/database" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"${DB}\"}" \
  || echo "Database poate exista deja — continui."

echo "InfluxDB init gata."
