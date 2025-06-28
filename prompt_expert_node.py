import requests
import json
import os
import re
import sys
from typing import Dict, List, Any, Optional

class PromptExpertNode:
    """
    一个ComfyUI节点，用于生翻译AI绘画提示词，具有专业提示词工程专家的能力。
    支持DeepSeek API和自定义Ollama API地址。输出干净的回答内容，无多余引号和think部分。
    将翻译和生成分开
    通过下拉框进行输入区分

    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_type": (["翻译", "提示词"],{"default": "翻译"}),
                "input_text": ("STRING", {"multiline": True, "default": "", "placeholder": "在这里输入您的问题或请求...", "dynamicPrompts": False}),
                "input_fanyi": ("STRING", {"multiline": True, "default": "你是个专业翻译，只翻译中文，直接翻译内容，单词翻译为单词，句子翻译为句子，不要组织语言,不要使用口语。丝袜翻译为stocking,肉色翻译为flesh-colored","tooltip": "当你输入包含翻译这两个字的时候，使用这个角色设定词", "dynamicPrompts": False}),
                "input_tishici": ("STRING", {"multiline": True, "default": "你是一个专业提供AI绘画提示词的专家。你擅长创作详细、能产生高质量图像的提示词,只使用英语回答，提示词之间使用,分开。不要使用口语,不要使用markdown格式。", "tooltip": "当你的输入没有翻译这两个字的时候，使用这个角色设定词", "dynamicPrompts": False}),
                "api_type": (["DeepSeek API", "Ollama API"], {"default": "Ollama API"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "DeepSeek API需要填写API密钥"}),
                "ollama_api_url": ("STRING", {"default": "http://localhost:11434", "multiline": False, "placeholder": "Ollama API地址,例如http://192.168.1.100:11434"}),
                "model_name": ("STRING", {"default": "deepseek-chat", "multiline": False, "placeholder": "模型名称,DeepSeek如deepseek-chat,Ollama如llama3"}),
                "target_language": (["中文", "English", "日本語", "한국어", "Français", "Deutsch", "Español", "Русский"], {"default": "English"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_prompt",)
    FUNCTION = "process_prompt"
    CATEGORY = "prompt tools"

    def __init__(self):
        self.output_dir = "outputs/prompt_expert"
        os.makedirs(self.output_dir, exist_ok=True)

    def process_prompt(self, input_text, input_fanyi, input_tishici, api_type, api_key, ollama_api_url, model_name, target_language):
        if target_type == "翻译" :
            system_prompt = input_fanyi
        else:
            system_prompt = input_tishici

        if not input_text.strip():
            user_message = f"请为我创建一个详细的AI绘画提示词,使用{target_language}语言。包含详细的主题、风格、光照、心情和构图描述。"
        else:
            user_message = input_text

        try:
            if api_type == "DeepSeek API":
                output = self._call_deepseek_api(api_key, model_name, system_prompt, user_message)
            else:
                output = self._call_ollama_api(ollama_api_url, model_name, system_prompt, user_message)

            clean_output = self._clean_output(output)
            translated_output = clean_output

            self._save_output(input_text, clean_output, translated_output)

            if '翻译' in input_text:
                return (translated_output,)
            else:
                return ("(masterpiece:1.0), (highest quality:1.12), (HDR:1.0), synchronization, detailed, realistic, 8k uhd, high quality " + clean_output,)

        except Exception as e:
            error_message = f"错误: {str(e)}"
            print(error_message)
            return (error_message,)

    def _call_deepseek_api(self, api_key, model_name, system_prompt, user_message):
        if not api_key.strip():
            raise ValueError("DeepSeek API需要API密钥。请在节点配置中提供有效的API密钥。")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions ",
            headers=headers,
            json=data
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"DeepSeek API错误: {response.text}")

    def _call_ollama_api(self, api_url, model_name, system_prompt, user_message):
        if not model_name.strip():
            raise ValueError("使用Ollama需要提供有效的模型名称。请在节点配置中提供模型名称。")
        if not api_url.strip():
            api_url = "http://localhost:11434"
        if not api_url.startswith("http"):
            api_url = "http://" + api_url
        api_url = api_url.rstrip("/")

        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": False
        }

        try:
            response = requests.post(
                f"{api_url}/api/chat",
                headers=headers,
                json=data,
                timeout=300
            )
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                raise Exception(f"Ollama API响应错误: 状态码 {response.status_code}, 响应: {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"无法连接到Ollama API ({api_url}): {str(e)}")

    def _clean_output(self, text):
        text = text.strip()
        text = re.sub(r'</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'[“”]', '', text)
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        elif text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        think_pattern = r'^think\s*:.*?(?=\n|\Z)'
        text = re.sub(think_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\n{3,}', '\n', text)
        return text.strip()

    def _save_output(self, input_text, output, translation=None):
        try:
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.output_dir, f"prompt_{timestamp}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write("=== 输入 ===\n")
                f.write(input_text + "\n")
                f.write("=== 输出 ===\n")
                f.write(output)
                if translation and translation != output:
                    f.write("\n=== 中文翻译 ===\n")
                    f.write(translation)
            print(f"已保存提示词到: {filename}")
        except Exception as e:
            print(f"保存提示词时出错: {str(e)}")


# 这部分对ComfyUI识别和加载节点是必要的
NODE_CLASS_MAPPINGS = {
    "PromptExpertNode": PromptExpertNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptExpertNode": "AI提示词专家"
}
