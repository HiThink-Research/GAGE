import json
import traceback
import aiohttp
import asyncio
from tqdm import tqdm
import pandas as pd
import base64
from PIL import Image
from io import BytesIO
import yaml
from openai import OpenAI
from loguru import logger
import copy
import sys,os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

logger.remove()
logger.add(sink=sys.stderr, level="INFO")

class AsyncLLMClient:
    def __init__(self, app_id, app_secret, url, chat_url, model, timeout=60,Semaphore=3):
        """
        初始化通用参数

        Args:
            app_id (str): 应用ID
            app_secret (str): 应用密钥
            url (str): 认证接口URL
            chat_url (str): 对话接口URL
            model (str): 使用的模型名称
            timeout (int): 请求超时时间，默认60秒
        """
        self.app_id = app_id
        self.app_secret = app_secret
        self.url = url
        self.chat_url = chat_url
        self.model = model
        self.timeout = timeout
        self.Semaphore = Semaphore
        self.authority = None
        self.session = None

    async def initialize(self):
        """初始化session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self

    async def close(self):
        """关闭session"""
        if self.session:
            await self.session.close()
            self.session = None

    @staticmethod
    def _encode_image(image_path):
        """
        将图片编码为base64格式
        
        Args:
            image_path (str): 图片的路径
            
        Returns:
            str: Base64编码的图片
        """
        pil_image = Image.open(image_path)
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format=pil_image.format)
        return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    
    async def refresh_authority(self):
        """刷新身份信息"""
        if self.session is None:
            await self.initialize()
            
        d = {"app_id": self.app_id, "app_secret": self.app_secret}
        h = {"Content-Type": "application/json"}
        
        async with self.session.post(self.url, json=d, headers=h, timeout=self.timeout) as response:
            data = await response.json()
            if data.get("success"):
                self.authority = data
            else:
                raise Exception(f"Failed to authenticate: {data}")

    @staticmethod
    def find_key_in_nested_dict(nested_dict, target_key):
        """
        嵌套字典中寻找目标Key

        Args:
            nested_dict (dict):嵌套字典
            target_key(str):要寻找的目标Key
        Returns:
            Dict: 如果找到返回键所在的值，否则返回 None
        """
        if not isinstance(nested_dict, dict):
            return None

        stack = [nested_dict]
        while stack:
            current_dict = stack.pop()
            if target_key in current_dict:
                return current_dict[target_key]
            for value in current_dict.values():
                if isinstance(value, dict):
                    stack.append(value)
        return None
    
    @staticmethod
    def save_file(path,instruction,predict_result):
        with open(path,'a',encoding='utf-8') as f: 
            tmp = instruction
            tmp['reasoning_content'] = predict_result['reasoning_content'] if isinstance(predict_result, dict) and 'reasoning_content' in predict_result else ""
            tmp['predict_result'] = predict_result['content'] if isinstance(predict_result, dict) and 'content' in predict_result else predict_result

            f.write(json.dumps(tmp,ensure_ascii=False)+"\n")
 
    async def get_gpt_response(self, messages, chat_url, tools=None, temperature=0, try_num=3, pbar=None):
        """
        大模型调用和身份验证

        Args:
            messages (Obj): messages模式请求格式
            chat_url(str): 对话接口URL
            temperature(float): 调用接口的温度
            try_num(int): retry最大次数
            pbar (tqdm): 进度条对象，可选
        Returns:
            Dict: 接口返回的text或者错误
        """
        if self.session is None:
            await self.initialize()

        error_messages = ""
        try:
            assert isinstance(messages, list) and not [x for x in messages if x["role"] not in {"system", "user", "assistant"}]
            chat_d = {
                "model": self.model,
                "messages": messages,
                "tools":tools,
                "temperature": temperature
            }
            if not self.authority:
                await self.refresh_authority()
            chat_h = {
                "Content-Type": "application/json",
                "userId": self.authority["data"]["user_id"],
                "token": self.authority["data"]["token"]
            }
        except Exception as e:
            error_messages = traceback.format_exc()
            logger.error(error_messages)
            return "error: "+error_messages

        for try_id in range(try_num):
            try:
                async with self.session.post(chat_url, json=chat_d, headers=chat_h, timeout=self.timeout) as response:
                    response_data = await response.json()
                    choices = self.find_key_in_nested_dict(response_data, 'choices')
                    if 'success' in response_data and not response_data.get('success'):
                        logger.error(chat_d)
                        logger.error("Retry time:{}\n{}".format(try_id,response_data))
                        continue
                    if choices is None:
                        logger.error("Retry time:{}\n{}".format(try_id,response_data))
                        continue
                    if pbar:
                        pbar.update(1)
                    logger.debug(response_data)
                    result = choices[0]["message"] if "reasoning_content" in choices[0]["message"] else choices[0]["message"]["content"]
                    return result
            except Exception as e:
                error_messages = traceback.format_exc()
                logger.error("Retry time:{}\n{}".format(try_id,error_messages))
                # await self.refresh_authority()
        return "error: "+str(error_messages)

    async def text2text(self, instruction, temperature=1,pbar=None,output_file=None):
        """
        大模型文本生成文本

        Args:
            instruction (str): input 文本
            temperature (float): 调用接口的温度
        Returns:
            Dict: 接口返回的text或者错误
        """
        if isinstance(instruction, str):
            messages = [
                {"role": "user", "content": instruction}
            ]
            instruction = {'messages':messages}
        else:
            messages = instruction['messages']
        tools = instruction['tools'] if 'tools' in instruction.keys() else None
        # print(instruction)
        predict_result = await self.get_gpt_response(messages, self.chat_url,tools,temperature=temperature,pbar=pbar)
        if output_file:
            self.save_file(output_file,instruction,predict_result)
        return predict_result
    
    async def image2text(self, instruction, image=None, temperature=0,pbar=None,output_file=None):
        """
        大模型图文生成文本

        Args:
            instruction(str): input 文本 或者message格式的list
            image (str): 图片路径，当传入message时可以是空
            temperature (float): 调用接口的温度
        Returns:
            Dict: 接口返回的text或者错误
        """
        if isinstance(instruction, str):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": instruction
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self._encode_image(image)}"
                            }
                        }
                    ]
                }
            ]
            instruction = {
                "messages":[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": instruction
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image
                                }
                            }
                        ]
                    }
                ]
            }
        else:
            messages = copy.deepcopy(instruction['messages'])
            for i,item in enumerate(messages):
                for j, item2 in enumerate(item["content"]):
                    if item2.get("type") == "image_url":
                        messages[i]["content"][j]["image_url"] = {
                            "url": f"data:image/jpeg;base64,{self._encode_image(item2['image_url']['url'])}"
                        }
        
        predict_result = await self.get_gpt_response(messages, self.chat_url, temperature,pbar=pbar)
        if output_file:
            self.save_file(output_file,instruction,predict_result)
        return predict_result

    async def images2texts(self, instructions: list, temperature=0, images: list=None,output_file=None):
        """
        大模型批量图文生成文本

        Args:
            instructions (list): input 一个列表，包含多个instruction,可以是messages格式
            images (list): 图片路径列表
            temperature (float): 调用接口的温度
        Returns:
            List: 接口返回一个结果列表，结果列表中元素顺序和instructions中元素顺序一致
        """
        await self.initialize()  # 确保session已初始化
        
        # 创建进度条
        pbar = tqdm(total=len(instructions), desc="Processing Image-Text Pairs")

        semaphore = asyncio.Semaphore(self.Semaphore)
        async def process_with_semaphore(instruction, image=None):
            async with semaphore:  # 使用信号量控制并发
                return await self.image2text(instruction, image, temperature, pbar=pbar, output_file=output_file)
    
        try:
            if images:
                tasks = [process_with_semaphore(instruction, image) 
                        for instruction, image in zip(instructions, images)]
            else:
                tasks = [process_with_semaphore(instruction) 
                        for instruction in instructions]
                
            # 执行所有任务
            results = await asyncio.gather(*tasks)
        finally:
            pbar.close()
            
        # 确保结果顺序与输入顺序一致
        return results

    async def texts2texts(self, instructions: list, temperature=1,output_file=None):
        """
        大模型批量文本生成文本

        Args:
            instructions (list): input 一个列表，包含多个instruction,可以是messages格式
            temperature (float): 调用接口的温度
        Returns:
            List: 接口返回一个结果列表，结果列表中元素顺序和instructions中元素顺序一致
        """
        await self.initialize()  # 确保session已初始化
        text_messages = []
        if not isinstance(instructions[0], str):
            for i,item in enumerate(instructions):
                if "messages" in item:
                    item = item['messages']
                tmp = []
                for text_item in item:
                    tmp.append({
                        "role": text_item["role"],
                        "content": "\n".join([x["text"] for x in text_item["content"]]) if isinstance(text_item["content"], list) else text_item["content"]
                    })
                instructions[i]['messages'] = tmp
        # print(text_messages)
        # 创建进度条
        pbar = tqdm(total=len(instructions), desc="Processing Text-to-Text")
        
        # 创建信号量来限制并发
        semaphore = asyncio.Semaphore(self.Semaphore)
        # import pdb;pdb.set_trace()
        # 创建一个包装函数，在其中使用信号量
        async def process_with_semaphore(message,i):
            async with semaphore:  # 使用信号量控制并发
                return await self.text2text(message, temperature=temperature, pbar=pbar, output_file=output_file)
        try:
            # 使用包装函数创建任务
            tasks = [process_with_semaphore(message,i) for i,message in enumerate(instructions)]
            
            # 执行所有任务
            results = await asyncio.gather(*tasks)
        finally:
            pbar.close()
        return results

class GPTClient:
    def __init__(self, api_name,api_key,model_name, 
                    base_url=None,
                    timeout=600, semaphore_limit=5):
        """
        初始化通用参数

        Args:
            api_name (str): API name e.g., chatgpt
            api_key (str): API 密钥
            model_name (str): 使用的模型名称
            base_url (str): 基础 URL
            timeout (int): 请求超时时间，默认 60 秒
            semaphore_limit (int): 并发限制，默认 5
        """
        self.api_name = api_name
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.semaphore_limit = semaphore_limit
        self.session = None

    async def initialize(self):
        """初始化异步会话"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        if self.api_name == 'chatgpt':
            os.environ["OPENAI_API_KEY"] = self.api_key
            self.client = OpenAI()
        return self

    async def close(self):
        """关闭异步会话"""
        if self.session:
            await self.session.close()
            self.session = None

    @staticmethod
    def _encode_image(image_path):
        """
        将图片编码为 Base64 格式

        Args:
            image_path (str): 图片路径

        Returns:
            str: Base64 编码的图片
        """
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format=img.format)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

    async def get_post_response(self, messages, chat_url,tools=None, temperature=0.01, try_num=3, pbar=None):
        """
        获取大模型响应

        Args:
            messages (list): 消息列表
            temperature (float): 温度参数
            try_num (int): 最大重试次数
            pbar (tqdm): 进度条对象（可选）

        Returns:
            dict: 接口返回的文本或错误信息
        """
        if self.session is None:
            await self.initialize()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "temperature": temperature
        }

        for attempt in range(try_num):
            try:
                async with self.session.post(chat_url, json=payload, headers=headers, timeout=self.timeout) as response:
                    response_data = await response.json()

                    if not response_data.get("success", True):
                        logger.error(f"Retry time: {attempt}, Error: {response_data}")
                        continue
                    # import pdb;pdb.set_trace()
                    choices = response_data['choices']
                    if choices is None:
                        logger.error("Openai error: {}".format(response_data))
                        continue
                    result = choices[0]['message']['content']
                    if pbar:
                        pbar.update(1)
                    return result
            except Exception as e:
                error_message = traceback.format_exc()
                logger.error(f"Retry time: {attempt}, Error: {error_message}, Openai response: {response_data}")
                # await self.refresh_authority()

        return {"error": "Max retries exceeded"}

    async def get_chatgpt_response(self, messages, temperature=0, try_num=3, pbar=None):
        """
        获取大模型响应

        Args:
            messages (list): 消息列表
            temperature (float): 温度参数
            try_num (int): 最大重试次数
            pbar (tqdm): 进度条对象（可选）

        Returns:
            dict: 接口返回的文本或错误信息
        """
        if self.session is None:
            await self.initialize()

        for attempt in range(try_num):
            try:
                response_data = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )

                choices = response_data.choices
                if choices is None:
                    logger.error("Openai error: {}".format(response_data))
                    continue
                result = choices[0].message.content
                if pbar:
                    pbar.update(1)
                return result
            except Exception as e:
                error_message = traceback.format_exc()
                logger.error(f"Retry time: {attempt}, Error: {error_message}, Openai response: {response_data}")

        return {"error": "Max retries exceeded"}

    async def text2text(self, instruction, temperature=0.01, pbar=None, output_file=None):
        """
        文本生成文本

        Args:
            instruction (str or list): 输入文本或消息列表
            temperature (float): 温度参数
            pbar (tqdm): 进度条对象（可选）
            output_file (str): 输出文件路径（可选）

        Returns:
            dict: 接口返回的文本或错误信息
        """
        if isinstance(instruction, str):
            messages = [
                {"role": "user", "content": instruction}
            ]
            instruction = {'messages':messages}
        else:
            messages = instruction['messages']
            tools = instruction['tools'] if 'tools' in instruction.keys() else None
        predict_result = await self.get_post_response(messages,tools,temperature=temperature,chat_url=self.base_url, try_num=2, pbar=pbar)
        # predict_result = await self.get_chatgpt_response(messages,temperature=temperature, try_num=2, pbar=pbar)

        if output_file:
            ins = instruction
            predict_result = predict_result
            self.save_file(output_file,ins,predict_result)
        
        return {'instruction':instruction,'predict_result':predict_result}

    async def texts2texts(self, instructions, temperature=0, output_file=None):
        """
        批量文本生成文本

        Args:
            instructions (list): 输入指令列表
            temperature (float): 温度参数
            output_file (str): 输出文件路径（可选）

        Returns:
            list: 接口返回的结果列表
        """
        await self.initialize()
        pbar = tqdm(total=len(instructions), desc="Processing Text-to-Text ({})".format(self.api_name))

        semaphore = asyncio.Semaphore(self.semaphore_limit)
        # import pdb;pdb.set_trace()
        async def process_with_semaphore(instruction):
            async with semaphore:
                return await self.text2text(instruction, temperature=temperature, pbar=pbar, output_file=output_file)

        tasks = [process_with_semaphore(instruction) for instruction in instructions]
        results = await asyncio.gather(*tasks)
        
        pbar.close()
        return results

    @staticmethod
    def save_file(path, instruction, predict_result):
        """
        保存结果到文件

        Args:
            path (str): 文件路径
            instruction (dict): 输入指令
            predict_result (dict): 预测结果
        """
        with open(path, 'a', encoding='utf-8') as f:
            tmp = instruction
            tmp['reasoning_content'] = predict_result.get('reasoning_content', "") if isinstance(predict_result, dict) else ""
            tmp['predict_result'] = predict_result.get('content', predict_result) if isinstance(predict_result, dict) else predict_result
            f.write(json.dumps(tmp, ensure_ascii=False) + "\n")


# 示例用法
if __name__ == "__main__":
    pass