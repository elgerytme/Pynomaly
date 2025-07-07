"""Celery tasks for long-running detection pipelines."""

from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def long_running_detection(data):
    # Simulate a long-running detection process
    result = process_detection(data)
    return result


def process_detection(data):
    # Placeholder processing logic
    return {"status": "completed", "result": "success"}
