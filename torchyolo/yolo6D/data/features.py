from datasets import Features, Value, Image
from jsonref import replace_refs
from embdata.sense.image import Image as MBImage
import logging

IDEFICS_FEATURES = Features({"messages": [{"role": Value('string'),"content": [{"type": Value('string'), "text": Value('string') }]}],
        "images": [Image()] })

PHI_FEATURES = Features({"messages": [{"role": Value('string'),"content": Value('string')}], "images": [Image()] })

from datasets import Features, Value

def json_schema_to_features(schema: dict) -> Features:
    """Convert a pedantic JSON schema to Hugging Face Datasets Features.

    Args:
        schema (dict): The pedantic JSON schema.

    Returns:
        Features: The converted Features object for Hugging Face Datasets.

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer"},
        ...         "scores": {
        ...             "type": "array",
        ...             "items": {"type": "integer"},
        ...             "shape": [3, 2]
        ...         }
        ...     }
        ... }
        >>> features = json_schema_to_features(schema)
        >>> features
        {'name': Value(dtype='string', id=None), 'age': Value(dtype='int32', id=None), 'scores': [[Value(dtype='int32', id=None)]]}
    """
    def convert_schema(schema_inner: dict):        
        schema_type = schema_inner.get("type")
        
        if schema_inner == MBImage(size=(256, 256)).schema():
            return Image()

        if schema_type == "string":
            return Value("string")
        elif schema_type == "integer":
            return Value("int32")
        elif schema_type == "number":
            return Value("float32")
        elif schema_type == "boolean":
            return Value("bool")
        elif schema_type == "array":
            items_schema = schema_inner.get("items", {})
            shape = schema_inner.get("shape", [])
            inner_feature = convert_schema(items_schema)
            
            # Create nested lists based on the shape
            if shape:
                for _ in shape:
                    inner_feature = [inner_feature]
            else:
                inner_feature = [inner_feature]  # Default to a single list if shape is not specified
            
            return inner_feature
        elif schema_type == "object":
            properties = schema_inner.get("properties", {})
            return Features({k: convert_schema(v) for k, v in properties.items()})
        else:
            logging.warning(f"Skipping unknown schema type: {schema_type}")

    if schema.get("type") != "object":
        raise ValueError("Schema must be of type 'object' at the root level.")

    return convert_schema(schema)

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)