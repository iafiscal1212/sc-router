"""Universal tool adapter.

Converts external tool definitions (functions, OpenAPI specs, descriptions)
into SC-Router Tool objects for use with the catalog.
"""

import inspect
import re
from typing import Any, Callable, Dict, List, Optional

from .catalog import Tool


class ToolAdapter:
    """Factory methods to create Tool objects from various sources."""

    @staticmethod
    def from_function(
        func: Callable,
        description: Optional[str] = None,
        capability_tags: Optional[set] = None,
        cost: float = 1.0,
    ) -> Tool:
        """Create a Tool from a Python function.

        Extracts input/output types from type hints and docstring.
        """
        name = func.__name__
        desc = description or func.__doc__ or f"Function: {name}"
        desc = desc.strip().split('\n')[0]  # first line only

        # Extract input types from signature
        sig = inspect.signature(func)
        input_types = set()
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                input_types.add(param.annotation.__name__
                                if hasattr(param.annotation, '__name__')
                                else str(param.annotation))
            else:
                input_types.add(param_name)

        # Extract output type from return annotation
        output_types = set()
        if sig.return_annotation != inspect.Signature.empty:
            ret = sig.return_annotation
            output_types.add(ret.__name__ if hasattr(ret, '__name__') else str(ret))

        # Auto-generate capability tags from name + description
        if capability_tags is None:
            words = re.findall(r'[a-z]+', name.lower().replace('_', ' '))
            desc_words = re.findall(r'[a-z]{3,}', desc.lower())
            capability_tags = set(words) | set(desc_words[:5])

        return Tool(
            name=name,
            description=desc,
            input_types=input_types,
            output_types=output_types,
            capability_tags=capability_tags,
            cost=cost,
        )

    @staticmethod
    def from_openapi(spec: Dict[str, Any]) -> List[Tool]:
        """Create Tools from an OpenAPI specification.

        Parses paths and operations to generate one Tool per endpoint.
        """
        tools = []
        paths = spec.get('paths', {})

        for path, methods in paths.items():
            for method, details in methods.items():
                if method.lower() in ('get', 'post', 'put', 'delete', 'patch'):
                    op_id = details.get('operationId', f"{method}_{path}")
                    summary = details.get('summary', details.get('description', ''))
                    tags = set(details.get('tags', []))

                    # Extract input types from parameters
                    input_types = set()
                    for param in details.get('parameters', []):
                        ptype = param.get('schema', {}).get('type', param.get('name', ''))
                        input_types.add(ptype)

                    # Request body
                    body = details.get('requestBody', {})
                    for content_type in body.get('content', {}):
                        input_types.add(content_type.split('/')[-1])

                    # Output types from responses
                    output_types = set()
                    for code, resp in details.get('responses', {}).items():
                        for ct in resp.get('content', {}):
                            output_types.add(ct.split('/')[-1])

                    # Capability tags from summary + tags
                    cap_tags = set(tags)
                    if summary:
                        words = re.findall(r'[a-z]{3,}', summary.lower())
                        cap_tags.update(words[:5])

                    tools.append(Tool(
                        name=op_id,
                        description=summary or f"{method.upper()} {path}",
                        input_types=input_types,
                        output_types=output_types,
                        capability_tags=cap_tags,
                    ))

        return tools

    @staticmethod
    def from_description(
        name: str,
        description: str,
        cost: float = 1.0,
    ) -> Tool:
        """Create a Tool from just a name and description.

        Auto-extracts capability tags from the description text.
        """
        words = re.findall(r'[a-z]{3,}', description.lower())
        # Filter common stop words
        stop = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'are', 'was',
                'will', 'can', 'has', 'have', 'been', 'not', 'but', 'its'}
        tags = {w for w in words if w not in stop}

        return Tool(
            name=name,
            description=description,
            input_types=set(),
            output_types=set(),
            capability_tags=tags,
            cost=cost,
        )
