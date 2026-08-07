import os
from contextlib import contextmanager

@contextmanager
def set_proxy():
    proxy_port = os.getenv("PROXY_PORT")
    if not proxy_port:
        yield
        return

    # Store original proxies
    originals = {
        'HTTP_PROXY': os.environ.get('HTTP_PROXY'),
        'HTTPS_PROXY': os.environ.get('HTTPS_PROXY'),
        'http_proxy': os.environ.get('http_proxy'),
        'https_proxy': os.environ.get('https_proxy'),
        'ALL_PROXY': os.environ.get('ALL_PROXY'),
        'all_proxy': os.environ.get('all_proxy')
    }

    proxy_url = f"http://127.0.0.1:{proxy_port}"
    
    # Set new proxies
    for key in originals.keys():
        os.environ[key] = proxy_url

    try:
        yield
    finally:
        # Restore original proxies
        for key, original_val in originals.items():
            if original_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_val
