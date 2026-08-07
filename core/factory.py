import importlib
from typing import Any, Dict, Optional, Type
from loguru import logger

class ComponentFactory:
    """
    Component factory for dynamically creating RAG components, 
    inspired by the modular design of the Bisheng project.
    """
    
    _registry = {
        "llm": {
            "ChatOpenAI": "langchain_openai.ChatOpenAI",
            "ChatZhipuAI": "langchain_community.chat_models.ChatZhipuAI",
        },
        "embeddings": {
            "HuggingFaceEmbeddings": "langchain_huggingface.HuggingFaceEmbeddings",
            "OpenAIEmbeddings": "langchain_openai.OpenAIEmbeddings",
        },
        "loaders": {
            "PyPDFLoader": "langchain_community.document_loaders.PyPDFLoader",
            "TextLoader": "langchain_community.document_loaders.TextLoader",
            "UnstructuredFileLoader": "langchain_community.document_loaders.UnstructuredFileLoader",
        },
        "splitters": {
            "RecursiveCharacterTextSplitter": "langchain_text_splitters.RecursiveCharacterTextSplitter",
        },
        "vectorstores": {
            "Chroma": "langchain_chroma.Chroma",
        }
    }

    @staticmethod
    def get_component(category: str, name: str, **kwargs) -> Any:
        """
        Dynamically import and instantiate a component.
        """
        try:
            if category not in ComponentFactory._registry:
                raise ValueError(f"Unknown category: {category}")
            
            class_path = ComponentFactory._registry[category].get(name)
            if not class_path:
                # Fallback to direct import if not in registry
                if "." in name:
                    class_path = name
                else:
                    raise ValueError(f"Component {name} not found in registry for category {category}")

            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            
            logger.info(f"Instantiating {category} component: {name}")
            return cls(**kwargs)
        except Exception as e:
            logger.error(f"Failed to load component {name} in {category}: {e}")
            raise

    @classmethod
    def register(cls, category: str, name: str, class_path: str):
        """Register a new component dynamically."""
        if category not in cls._registry:
            cls._registry[category] = {}
        cls._registry[category][name] = class_path
