import unittest
from httpx import AsyncClient

class TestGenerateChat(unittest.IsolatedAsyncioTestCase):
    """
    测试生成聊天内容
    1. 先启动服务
    ```bash
    python api.py
    ```
    2. 运行测试
    ```bash
    python -m unittest api_test.py
    ```
    """
    async def test_generate_chat(self):
        async with AsyncClient() as ac:
            response = await ac.post(
                "http://localhost:8000/",
                json={
                    "prompt": "你好",
                    "history": [],
                    "max_length": 2048,
                    "top_p": 0.7,
                    "temperature": 0.95
                })
        self.assertEqual(response.status_code, 200)
        print(response.json())