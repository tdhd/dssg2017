#!/bin/bash

PORT=$(grep -Po "PORT = \K([0-9]*)$" dkg/settings.py)
echo $PORT
while true; do
  echo "Re-starting Django runserver"
  python manage.py runserver 0.0.0.0:$PORT
  sleep 2
done

