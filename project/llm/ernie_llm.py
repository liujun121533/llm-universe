#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   wenxin_llm.py
@Time    :   2023/10/16 18:53:26
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于百度文心大模型自定义 LLM 类
'''

from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
from pydantic import Field
from llm.self_llm import Self_LLM
import json
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun

import os
from dotenv import find_dotenv, load_dotenv

# 调用文心 API 的工具函数
def get_access_token(api_key : str, secret_key : str):
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
    # 指定网址
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    # 设置 POST 访问
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    # 通过 POST 访问获取账户对应的 access_token
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

class Ernie_LLM(Self_LLM):
    # 文心大模型的自定义 LLM
    # URL
    url : str = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={}"
    # Secret_Key
    secret_key : str = None
    # access_token
    access_token: str = None

    def init_access_token(self):
        try:
            _ = load_dotenv(find_dotenv())

            self.access_token = os.environ["EB_AGENT_ACCESS_TOKEN"]
        except Exception as e:
            print(e)
            print("获取 access_token 失败，请检查 Key")
        

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        self.init_access_token()
        import erniebot
        erniebot.api_type = "aistudio"
        erniebot.access_token = self.access_token

        stream = False
        response = erniebot.ChatCompletion.create(
            model="ernie-3.5",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=self.temperature,
            stream=stream)

        result = "请求失败"
        if stream:
            for resp in response:
                result += resp.get_result()
        else:
            result = response.get_result()
        return result
        
        
    @property
    def _llm_type(self) -> str:
        return "Ernie"
