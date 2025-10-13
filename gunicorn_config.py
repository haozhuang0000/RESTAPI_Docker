# gunicorn_config.py
from gevent import monkey
monkey.patch_all()

# Gunicorn config variables
bind = "0.0.0.0:7777"
worker_class = "gevent"
workers = 4  # Adjust based on your CPU cores
worker_connections = 2000
timeout = 300  # 5 minutes
keepalive = 75
max_requests = 10000
max_requests_jitter = 1000

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Performance tuning
backlog = 4096
graceful_timeout = 300
threads = 4
preload_app = True

def on_starting(server):
    print("Starting Gunicorn server with gevent worker class")

def on_exit(server):
    print("Shutting down Gunicorn server")