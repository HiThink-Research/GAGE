import json
import requests
import urllib.parse
from typing import Dict, Optional, Union
from urllib.error import URLError, HTTPError
from loguru import logger

class HTTPRequest:
    """
    一个 HTTP 请求封装类
    """
    def __init__(self, base_url: str = "", timeout: int = 30):
        """
        初始化请求类
        
        Args:
            base_url (str): API 的基础 URL
            timeout (int): 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Python HTTP Client/1.0',
            'Content-Type': 'application/json'
        }

    def set_header(self, key: str, value: str) -> None:
        """
        设置请求头
        
        Args:
            key (str): 头部字段名
            value (str): 头部字段值
        """
        self.headers[key] = value

    def _build_url(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """
        构建完整的 URL
        
        Args:
            endpoint (str): API 端点
            params (Dict, optional): URL 参数
            
        Returns:
            str: 完整的 URL
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if params:
            query_string = urllib.parse.urlencode(params)
            url = f"{url}?{query_string}"
        return url

    def _make_request(self, method: str, endpoint: str,
                    params: Optional[Dict] = None,
                    data: Optional[Union[Dict, str]] = None,
                    files: Optional[Dict] = None) -> Dict:
        """
        发送 HTTP 请求

        Args:
            method (str): HTTP 方法
            endpoint (str): API 端点
            params (Dict, optional): URL 参数
            data (Union[Dict, str], optional): 请求体数据
            files (Dict, optional): 要上传的文件

        Returns:
            Dict: 响应数据

        Raises:
            HTTPError: 当服务器返回错误状态码时
            ConnectionError: 当网络连接出现问题时
        """
        url = self._build_url(endpoint, params)

        # 准备请求数据
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                data = data

        try:
            if files:
                response = requests.request(
                    method=method,
                    url=url,
                    files=files,
                    timeout=self.timeout
                )
            else:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=data if isinstance(data, dict) else None,
                    data=data if isinstance(data, str) else None,
                    timeout=self.timeout
                )
            # 确保请求成功
            response.raise_for_status()
            
            # 尝试返回 JSON 数据
            try:
                return response.json()
            except ValueError:
                return {} if not response.text else {"response": response.text}
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e.response.status_code} - {e.response.reason}:{url}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection Error: {str(e)}")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout Error: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Unexpected Error: {str(e)}")
            raise

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        发送 GET 请求
        
        Args:
            endpoint (str): API 端点
            params (Dict, optional): URL 参数
            
        Returns:
            Dict: 响应数据
        """
        return self._make_request('GET', endpoint, params=params)

    def post(self, endpoint: str, data: Optional[Union[Dict, str]] = None, 
             params: Optional[Dict] = None,
             files: Optional[str] = None) -> Dict:
        """
        发送 POST 请求
        
        Args:
            endpoint (str): API 端点
            data (Union[Dict, str]): 请求体数据
            params (Dict, optional): URL 参数
            files (Src, optional): 传输的文件
            
        Returns:
            Dict: 响应数据
        """
        return self._make_request('POST', endpoint, params=params, data=data,files=files)

# 使用示例
if __name__ == "__main__":
    # 创建请求实例
    client = HTTPRequest("http://iwc-index-mobilesearch-156400:8080")
    
    # 设置认证头
    # client.set_header("Authorization", "Bearer your-token-here")
    
    try:
        # GET 请求示例
        # response = client.get("/users", {"page": 1, "limit": 10})
        # print("GET Response:", response)
        
        # POST 请求示例
        user_data = {
            "query": "Elon Musk and trump",
            "qid": "123455443322",
            "app_id": "test",
            "user_id": "test2",
            "channel": "news_en",
            "size": 10,
            "slots": [
                {
                "type": "time",
                "time_start": 1727665175,
                "time_end": 1727665175
                },
                {
                "type": "sentiment",
                "name": "Negative"
                },
                {
                "type": "stock",
                "values": [
                    "300033.SZ",
                    "000001.SH"
                ]
                },
                {
                "type": "industry",
                "values": [
                    "801095",
                    "801095"
                ]
                },
                {
                "type": "concept",
                "values": [
                    "净资产收益率roe"
                ]
                },
                {
                "type": "organization",
                "values": [
                    "华鑫证券",
                    "东北证券"
                ]
                },
                {
                "type": "host",
                "values": [
                    "www.163.com",
                    "www.qq.com"
                ]
                },
            {
                "type": "exclude_host",
                "values": [
                    "www.163.com"
                ]
                },
                {
                "type": "source",
                "values": [
                    "山东财经报道",
                    "东北证券"
                ]
                }
            ]
            }
        response = client.post("/", user_data)
        logger.error("POST Response:", response)
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")