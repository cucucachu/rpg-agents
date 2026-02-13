"""CORS configuration for production deployment.

This module provides production-ready CORS configuration that restricts
API access to specific origins while allowing development flexibility.
"""

import os
from typing import List


def get_allowed_origins() -> List[str]:
    """
    Get list of allowed CORS origins.
    
    In production, restricts to the actual domain.
    In development, allows localhost.
    
    Environment Variables:
        ALLOWED_ORIGINS: Comma-separated list of allowed origins
        DEBUG: If "true", adds localhost:3000
    
    Returns:
        List of allowed origin URLs
    """
    # Get from environment (production)
    env_origins = os.getenv("ALLOWED_ORIGINS", "")
    
    if env_origins:
        origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()]
    else:
        # Default production origins
        origins = [
            "https://conjurerstable.com",
            "https://www.conjurerstable.com",
        ]
    
    # Add development origins if DEBUG mode
    if os.getenv("DEBUG", "false").lower() == "true":
        origins.extend([
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ])
    
    return origins


def should_allow_credentials() -> bool:
    """
    Determine if credentials (cookies, auth headers) should be allowed.
    
    Returns:
        True to allow credentials (required for JWT auth)
    """
    return True


def get_allowed_methods() -> List[str]:
    """
    Get list of allowed HTTP methods.
    
    Returns:
        List of allowed HTTP methods
    """
    return ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


def get_allowed_headers() -> List[str]:
    """
    Get list of allowed request headers.
    
    Returns:
        List of allowed headers
    """
    return [
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "User-Agent",
        "DNT",
        "Cache-Control",
        "X-Requested-With",
    ]


def get_exposed_headers() -> List[str]:
    """
    Get list of headers that browsers can access.
    
    Returns:
        List of exposed headers
    """
    return [
        "Content-Length",
        "Content-Type",
        "X-Total-Count",
    ]


# Configuration dictionary for FastAPI middleware
CORS_CONFIG = {
    "allow_origins": get_allowed_origins(),
    "allow_credentials": should_allow_credentials(),
    "allow_methods": get_allowed_methods(),
    "allow_headers": get_allowed_headers(),
    "expose_headers": get_exposed_headers(),
    "max_age": 600,  # Cache preflight requests for 10 minutes
}


def print_cors_config():
    """Print current CORS configuration for debugging."""
    print("CORS Configuration:")
    print(f"  Allowed Origins: {CORS_CONFIG['allow_origins']}")
    print(f"  Allow Credentials: {CORS_CONFIG['allow_credentials']}")
    print(f"  Allowed Methods: {CORS_CONFIG['allow_methods']}")
    print(f"  Allowed Headers: {CORS_CONFIG['allow_headers']}")
    print(f"  Max Age: {CORS_CONFIG['max_age']}s")


if __name__ == "__main__":
    print_cors_config()
