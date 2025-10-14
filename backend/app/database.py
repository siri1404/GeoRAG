from supabase import create_client, Client
from app.config import get_settings
from functools import lru_cache

settings = get_settings()

@lru_cache()
def get_supabase() -> Client:
    return create_client(settings.supabase_url, settings.supabase_key)

@lru_cache()
def get_supabase_admin() -> Client:
    return create_client(settings.supabase_url, settings.supabase_service_role_key)
