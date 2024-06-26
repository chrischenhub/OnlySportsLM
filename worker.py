import requests
import json

# 定义服务器的URL
SERVER_URL = "http://120.26.210.154"

def get_task():
    # 发送获取任务的请求
    response = requests.post(f"{SERVER_URL}/getTask")
    if response.status_code == 200:
        task = response.json()
        print("Received task:", task)
        return task['task']
    else:
        print("Failed to get task:", response.text)
        return None

def update_task(task_name):
    # 发送更新任务状态的请求
    data = {
        "name": "worker1",  # 假设worker的名字是worker1
        "task": task_name,
        "status": 2
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f"{SERVER_URL}/updateTask", data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        print("Task updated:", response.text)
    else:
        print("Failed to update task:", response.text)

def main():
    # 获取任务
    task_name = get_task()
    if task_name:
        # 更新任务状态
        update_task(task_name)

if __name__ == "__main__":
    main()
